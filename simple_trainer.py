import torch
from utils.utils import AverageMeter, plot_graph
import time, os, shutil
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, loader, optimizer, scaler, loss_func, args):
    model.train()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data['image'].to(args.device), batch_data['label'].to(args.device)
        for param in model.parameters():
            param.grad = None
        if args.amp:
            with torch.amp.autocast('cuda'):
                logits = model(data)
                loss = loss_func(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(data)
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
        run_loss.update(loss.item(), n=1)
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, acc_func, args, model_inferer=None, post_label=None, post_pred=None, loss_func=None):
    model.eval()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    organ_dice =  DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data['image'].to(args.device), batch_data['label'].to(args.device)

            if args.amp:
                with torch.amp.autocast('cuda'):
                    if model_inferer is not None:
                        logits = model_inferer(data)
                    else:
                        logits = model(data)
                    loss = loss_func(logits, target)
            else:
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
                loss = loss_func(logits, target)
            run_loss.update(loss.item(), n=1)

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            # per class dice score
            organ_dice(y_pred=val_output_convert, y=val_labels_convert)

        metric_org = organ_dice.aggregate()
        organ_dice.reset()
        print("Dice accuracy for each class: ", metric_org)


    return run_acc.avg, run_loss.avg


def save_checkpoint(model, epoch, args, filename='model.pt', best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def simple_fit(model, train_loader, val_loader, optimizer, loss_func, acc_func, args, model_inferer=None,
               scheduler=None, start_epoch=0, post_label=None, post_pred=None, folder=None, num_model=0):

    writer = None
    if args.logdir is not None:
        writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'logs'))
        print("Writing Tensorboard logs to ", os.path.join(args.logdir, 'logs'))
    scaler = None
    if args.amp:
        scaler = torch.amp.GradScaler('cuda')

    val_acc_max = 0.
    trigger_times = 0
    patience = args.patience

    for epoch in range(start_epoch, args.max_epochs):
        epoch_time = time.time()
        torch.cuda.empty_cache()  # PyTorch thing
        train_loss = train_epoch(model, train_loader, optimizer, scaler=scaler, loss_func=loss_func,args=args)
        print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
              'time {:.2f}s'.format(time.time() - epoch_time))
        if writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        torch.cuda.empty_cache()  # PyTorch thing

        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_avg_acc, val_loss = val_epoch(model, val_loader, acc_func=acc_func, model_inferer=model_inferer,
                                              args=args, post_label=post_label, post_pred=post_pred, loss_func=loss_func)

            # losses[0].append(train_loss)
            # losses[1].append(val_loss)
            if writer is not None:
                writer.add_scalar("dice_loss/train", train_loss, epoch)
                writer.add_scalar("dice_loss/val", train_loss, epoch)

            print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                  'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time))

            if val_avg_acc > val_acc_max:
                print('Reset trigger time to 0')
                trigger_times = 0
                print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc.item()))
                val_acc_max = val_avg_acc.item()
                b_new_best = True
                if args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(model, epoch, args,
                                    best_acc=val_acc_max,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    filename=f"{num_model}model_lastsaved_fold{folder}.pt"
                                    )

                    if b_new_best:
                        print('Copying to model.pt new best model!!!!')
                        shutil.copyfile(os.path.join(args.logdir, f"{num_model}model_lastsaved_fold{folder}.pt"),
                                        os.path.join(args.logdir, f"{num_model}model_fold{folder}.pt"))
            else:
                trigger_times += 1
                print(f'trigger times: {trigger_times}')
                if trigger_times >= patience:
                    print('Early stopping!')
                    break

        if scheduler is not None:
            scheduler.step()

    if args.val_every > args.max_epochs:
        val_acc_max, _ = val_epoch(model, train_loader, acc_func=acc_func, model_inferer=model_inferer, args=args,
                                   post_label=post_label, post_pred=post_pred, loss_func=loss_func)
        if args.logdir is not None and args.save_checkpoint:
            save_checkpoint(model, args.max_epochs, args,
                            best_acc=val_acc_max,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            filename=f"{num_model}model_lastsaved_fold{folder}.pt"
                            )

    # plot_graph(losses, 'Epochs', ['Train Loss', 'Val. Loss'], savename=os.path.join(args.logdir, f"train-val_loss_folder{folder}"))
    print('Training Finished !, Best Accuracy: ', val_acc_max)

    return val_acc_max

import os
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from functools import partial
import nibabel as nib

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete, Activations, Compose
from monai.utils import set_determinism
from monai.networks.nets import DynUNet

from networks.sshunet import GateDynUNet
from utils.args import get_main_args
from utils.data_utils import get_loader, get_kernels_strides
from utils.utils import DeepDiceLoss
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from simple_trainer import simple_fit

torch.backends.cudnn.benchmark=True
nib.imageglobals.logger.setLevel(40)


def main(num_model=0):
    args = get_main_args()
    if torch.cuda.is_available():
        args.amp = True
        args.device = "cuda"
    else:
        args.amp = False
        args.device = "cpu"

    main_worker(args=args, num_model=num_model)


def main_worker(args, num_model):
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.test_mode = False
    args.val_mode = False
    print(f"------------------------------TRAINING OF MODEL {num_model+1}------------------------------")
    print(f"You are using the following device: {args.device}")
    print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    assert args.roi_x == args.roi_y == args.roi_z, "Patch size must be equal for each dimension!"
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    inf_space = [args.space_x, args.space_y, args.space_z]
    pretrained_dir = args.pretrained_dir

    accuracies = []
    num = args.nfolds
    folds = list(range(num))

    for folder in range(num):
        print(f"--------------------Folder {folder}-------------------\n")
        kernels, strides = get_kernels_strides(inf_size, inf_space)
        if args.kernels is not None:
            kernels = args.kernels
        else:
            print("CAREFUL, you didn't set the kernel size you are using fully 3d convolution!")
        if args.strides is not None:
            strides = args.strides

        if args.gate_type == None:
            print("You are doing MONAI implementation")
            model = DynUNet(
                spatial_dims=args.spatial_dims,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                kernel_size=kernels,
                strides=strides,
                upsample_kernel_size=strides[1:],
                filters=args.filters,
                dropout=args.dropout_rate,
                norm_name=args.norm_name,
                deep_supervision=args.deep_supervision,
                deep_supr_num=args.deep_supr_num,
                res_block=args.res_block,
                trans_bias=True, )
        else:
            model = GateDynUNet(
                spatial_dims=3,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                gate_type = args.gate_type,
                gate_pos=args.gate_pos,
                gate_in_bottleneck=args.gate_bottleneck,
                gate_dec=args.gate_dec,
                do_basic=False,
                kernel_size=kernels,
                strides=strides,
                upsample_kernel_size=strides[1:],
                filters=args.filters,
                dropout=args.dropout_rate,
                norm_name=args.norm_name,
                deep_supervision=args.deep_supervision,
                deep_supr_num=args.deep_supr_num,
                vfn=args.vfn,
                trans_bias=True, )

        if args.resume_weights_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))["state_dict"]
            model.load_state_dict(model_dict)
            print("Use pretrained weights")

        if args.out_channels > 2:
            print("Cross Entropy Dice Loss")
            dice_loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
        else :
            dice_loss = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)

        # dice_loss = DeepDiceLoss(dice_loss, args.deep_supervision, args.deep_supr_num)

        post_label = AsDiscrete(to_onehot=args.out_channels)
        post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=args.out_channels)])
        dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)   # MetricReduction.MEAN

        model_inferer = partial(
            sliding_window_inference,
            roi_size= inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
        )

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters count", pytorch_total_params)
        print(f"Filters: {model.filters},\nKernels: {kernels}\nStrides: {strides}")
        print(model)

        best_acc = 0
        start_epoch = 0
        if args.checkpoint is not None:
            checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            model.load_state_dict(new_state_dict, strict=False)
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            if "best_acc" in checkpoint:
                best_acc = checkpoint["best_acc"]
            print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

        model.to(args.device)

        if args.optim_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        elif args.optim_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        elif args.optim_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True,
                weight_decay=args.reg_weight
            )
        else:
            raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

        if args.lrschedule == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
            )
        elif args.lrschedule == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
            if args.checkpoint is not None:
                scheduler.step(epoch=start_epoch)
        else:
            scheduler = None

        if args.nfolds > 1:
            print("ENABLE K-fold cross-validation")
            train_folds = folds[0: folder] + folds[(folder + 1):]
            val_folds = folds[folder]
            loader = get_loader(args, train_folds, val_folds)
        else:
            loader = get_loader(args)

        accuracy = simple_fit(
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc,
            args=args,
            model_inferer=model_inferer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
            folder=folder,
            num_model=num_model
        )
        accuracies.append(accuracy)
    print("\n\nTraining Finished !, Best mean Validation Accuracy: ", np.mean(accuracies))
    return


if __name__ == '__main__':
    args = get_main_args()
    if args.num_models == 1:
        set_determinism(seed=4294967295)
    for num_model in range(args.num_models):
        main(num_model)

import argparse


def list_of_int(alist):
    newlist = []
    sublist = [int(element) for element in alist]
    newlist.extend(sublist)
    return newlist


def non_negative_int(value):
    ivalue = int(value)
    assert ivalue >= 0, f"Argparse error. Expected non-negative integer but got {value}"
    return ivalue


def list_of_float(alist):
    return [float[i] for i in alist]


def get_main_args(strings=None):
    parser = argparse.ArgumentParser(description="SSH-UNet segmentation pipeline")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="start training from saved checkpoint, insert the directory")
    parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
    parser.add_argument("--resume_weights_ckpt", action="store_true",
                        help="resume only model weights, ensure --pretrained_dir, --pretrained_model_name are correctly set")
    parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str,
                        help="pretrained checkpoint directory")
    parser.add_argument("--pretrained_model_name", default="0model_lastsaved_fold0.pt", type=str,
                        help="pretrained model name")
    parser.add_argument("--logdir", default="test", type=str,
                        help="directory to save the tensorboard logs and model weights")
    parser.add_argument("--data_dir", default="/data/cugwu/data/btcv/Training", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="dataset.json", type=str,
                        help="dataset json file, ensure the json is in the same folder of the dataset")
    parser.add_argument("--out_dir", default="/data/cugwu/data/msd/test_pred", type=str,
                        help="output directory for test and validation")
    parser.add_argument("--exp_name", default="test05", type=str, help="experiment name for test and validation")
    parser.add_argument("--mri", action="store_true", help="using MRI data transforms otherwise CT scans")

    parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
    parser.add_argument("--optim_lr", default=4e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str,
                        help="optimization algorithm, (options: 'adamw', 'adam', 'sgd')")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--val_every", default=100, type=int,
                        help="validation frequency, if bigger of --mas_epochs only validation is computed")
    parser.add_argument("--patience", default=10, type=int, help="patience for early stopping")
    parser.add_argument("--workers", default=1, type=int, help="number of workers")
    parser.add_argument("--nfolds", default=1, type=int, help='folds for k-fold cross validation')
    parser.add_argument("--num_models", default=1, type=int, help='number of models to train for ensamble')

    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    parser.add_argument("--gate_pos", default=-1, type=int, nargs='+', help="Gate positions")
    parser.add_argument("--gate_type", default=None, type=str,
                        help="choose the type of shifting module (tsm, gsm, gsf), if None you are using DynUnet from MONAI")
    parser.add_argument("--spatial_dims", default=3, type=int, help="cernel dimension of DynUnet of MONAI")
    parser.add_argument("--filters", default=None, type=int, nargs='+', help="list of filters to use")
    parser.add_argument("--gate_bottleneck", action='store_true', help="enables shifting in the bottleneck")
    parser.add_argument("--gate_dec", action='store_true', help="enables shifting on the decoder")
    parser.add_argument("--do_basic", action="store_true", help="enables Basic Block on the decoder")
    parser.add_argument("--kernels", type=lambda x: list_of_int(x), default=None, nargs='+',
                        help="convolution kernel size")
    parser.add_argument("--strides", type=lambda x: list_of_int(x), default=None, nargs='+',
                        help="convolution stride for each block")
    parser.add_argument("--res_block", action="store_true", help="use residual blocks for dynamic unet")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
    parser.add_argument("--deep_supr_num", type=non_negative_int, default=2, help="number of deep supervision heads")
    parser.add_argument("--deep_supervision", action="store_true", help="enable deep supervision")
    parser.add_argument("--vfn", action="store_true", help="enable VFN on the output block")

    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=275.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--pos", default=1, type=int,
                        help="used with neg together to calculate the ratio pos / (pos + neg) for the probability to pick a foreground voxel as a center rather than a background voxel.")
    parser.add_argument("--neg", default=1, type=int,
                        help="used with neg together to calculate the ratio pos / (pos + neg) for the probability to pick a foreground voxel as a center rather than a background voxel.")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.2, type=float,
                        help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float,
                        help="RandShiftIntensityd aug probability")
    parser.add_argument("--cache_num", default=30, type=int, help="number of sample to cache")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("--lrschedule", default="cosine_anneal", type=str,
                        help="type of learning rate scheduler (options: warmup_cosine, cosine_anneal) otherwise no scheduler is applied")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
    parser.add_argument("--ensemble_method", default="mean", type=str, choices=['mean', 'vote'], help="ensemble method")
    parser.add_argument("--w_mee", nargs='+', type=float, help="weights for Mean Ensemble Evaluation")

    if strings is not None:
        parser.add_argument(
            "strings",
            metavar="STRING",
            nargs="*",
            help="String for searching",
        )
        args = parser.parse_args(strings.split())
    else:
        args = parser.parse_args()
    return args

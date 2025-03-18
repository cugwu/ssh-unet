# https://github.com/Project-MONAI/tutorials/blob/main/modules/cross_validation_models_ensemble.ipynb
import os, sys, logging
import nibabel as nib
import torch

from networks.sshunet import GateDynUNet
from utils.args import get_main_args
from utils.data_utils import get_loader, get_kernels_strides

from monai.inferers import SlidingWindowInferer
from monai.networks.nets import DynUNet
from monai.handlers import MeanDice, from_engine, HausdorffDistance, SurfaceDistance
from monai.transforms import (
    AsDiscreted,
    Compose,
    MeanEnsembled,
    EnsureTyped,
    Activationsd,
    VoteEnsembled,
    SaveImaged
)
from monai.engines import EnsembleEvaluator
from monai.utils import set_determinism, first


nib.imageglobals.logger.setLevel(40)


def ensemble_evaluate(post_transforms, models, pred_keys, device, loader, inferer):
    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=loader,
        pred_keys=pred_keys,
        networks=models,
        inferer=inferer,
        postprocessing=post_transforms,
        key_val_metric={
            "test_mean_dice": MeanDice(
                include_background=True,
                reduction='mean_batch',
                output_transform=from_engine(["pred", "label"])
            ),
        },
    )
    evaluator.run()

def main(args):
    args.out_dir = os.path.join(args.out_dir, args.exp_name)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.test_mode = False
    args.val_mode = False
    args.nfolds = -1
    _, test_loader = get_loader(args)

    set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    inf_space = [args.space_x, args.space_y, args.space_z]
    kernels, strides = get_kernels_strides(inf_size, inf_space)
    pred_keys = [f'pred{i}' for i in range(args.num_models)]
    models = []
    for num in range(args.num_models):
        model_name = f'{0}{args.pretrained_model_name}{num}.pt'
        # model_name = args.pretrained_model_name

        pretrained_dir = args.pretrained_dir
        pretrained_pth = os.path.join(pretrained_dir, model_name)
        print(pretrained_pth)
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
            if args.kernels is not None:
                kernels = args.kernels
            if args.strides is not None:
                strides = args.strides
            model = GateDynUNet(
                spatial_dims=3,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                gate_type=args.gate_type,
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

        model_dict = torch.load(pretrained_pth, map_location=torch.device(device))
        model.load_state_dict(model_dict['state_dict'])
        model.eval()
        model.to(device)
        models.append(model)
    # ---------------------------- ENSEMBLE EVALUATION ---------------------------- #

    test_inferer = SlidingWindowInferer(roi_size=(args.roi_x, args.roi_y, args.roi_z), sw_batch_size=args.sw_batch_size,
                                        overlap=args.infer_overlap)
    if args.ensemble_method == 'mean':
        w_mee = args.w_mee
        if len(w_mee) != len(pred_keys):
            raise ValueError("You should choose a number of weights equal to num_models")
        mean_post_transforms = Compose(
            [
                EnsureTyped(keys=pred_keys),
                Activationsd(keys=pred_keys, softmax=True),
                AsDiscreted(keys=pred_keys, argmax=True, threshold=0.5),
                MeanEnsembled(keys=pred_keys, output_key="pred", weights=w_mee),
                SaveImaged(
                    keys="pred",
                    meta_keys="image_meta_dict",
                    output_ext=".nii.gz",
                    output_dir=args.out_dir,
                    output_postfix="",
                    output_dtype="uint8",
                    separate_folder=False,
                    resample=False,
                ),
            ]
        )
        torch.cuda.empty_cache()
        ensemble_evaluate(mean_post_transforms, models, pred_keys, torch.device(device), test_loader, test_inferer)
        print("Mean ensemble finished")

    else:
        vote_post_transforms = Compose(
            [
                EnsureTyped(keys=pred_keys),
                Activationsd(keys=pred_keys, softmax=True),
                AsDiscreted(keys=pred_keys, argmax=True, threshold=0.5),
                VoteEnsembled(keys=pred_keys, output_key="pred"),
                SaveImaged(
                    keys="pred",
                    meta_keys="image_meta_dict",
                    output_ext=".nii.gz",
                    output_dir=args.out_dir,
                    output_postfix="",
                    output_dtype="uint8",
                    separate_folder=False,
                    resample=False,
                ),
            ]
        )
        torch.cuda.empty_cache()
        ensemble_evaluate(vote_post_transforms, models, pred_keys, torch.device(device), test_loader, test_inferer)
        print("Vote ensemble finished")


if __name__ == "__main__":
    args = get_main_args()
    main(args)

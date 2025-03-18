import os
import time
import nibabel as nib
import torch.cuda.nvtx as nvtx
import torch
import numpy as np
import matplotlib.pyplot as plt

from networks.sshunet import GateDynUNet
from utils.args import get_main_args
from utils.data_utils import get_loader, get_kernels_strides

from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from monai.data import decollate_batch
from monai.handlers import from_engine
from monai.metrics import DiceMetric, SurfaceDiceMetric


nib.imageglobals.logger.setLevel(40)

def main(args):
    pretrained_pth = os.path.join(args.pretrained_dir, args.pretrained_model_name)

    args.test_mode = False
    args.val_mode = True
    val_loader, post_transforms = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inf_size = (args.roi_x, args.roi_y, args.roi_z)
    inf_space = (args.space_x, args.space_y, args.space_z)
    kernels, strides = get_kernels_strides(inf_size, inf_space)
    if args.kernels is not None:
        kernels = args.kernels
    if args.strides is not None:
        strides = args.strides

    if args.gate_type == None:
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

    model_dict = torch.load(pretrained_pth, map_location=torch.device(device))["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    del model_dict

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)
    inference = []
    memory_allocated = []
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)

            torch.cuda.empty_cache()

            initial_memory = torch.cuda.memory_allocated(device=device)
            start = time.time()
            val_data["pred"] = sliding_window_inference(val_inputs, inf_size, 4, model,
                                                        overlap=args.infer_overlap, mode="gaussian")
            inference.append(time.time() - start)

            final_memory = torch.cuda.memory_allocated(device=device)
            memory_allocated.append(final_memory - initial_memory)

            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate() # .item() # senza reduction ho la media generale
        # reset the status for next validation round
        dice_metric.reset()
    print("Avarage inference time: ", np.mean(inference))
    print(f"Metric on original image spacing: {metric_org}")
    print(f"Memory Allocated: {np.mean(memory_allocated) / (1024 ** 3)} GB")

if __name__ == "__main__":
    args = get_main_args()
    main(args)

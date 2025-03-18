# Source Code: https://github.com/Project-MONAI/tutorials/blob/main/modules/transform_visualization.ipynb
from monai.utils import first
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    AsDiscreted,
    AsDiscrete,
    ToTensord,
    LoadImage,
    SaveImage
)
from monai.data import DataLoader, Dataset
import os
import glob

import matplotlib.pyplot as plt

import numpy as np
import torch
import nibabel as nib

from skimage.morphology import erosion, disk

if __name__ == "__main__":
    data_dir = '/data/vision_group/medical/btcv/'
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs/img0062.nii.gz")))
    # train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsVa/amos_0029.nii.gz")))
    train_preds = sorted(glob.glob(os.path.join(data_dir, "results_tsm/val_pred/img0062.nii.gz")))

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_preds)
    ]

    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="PLS"),
            ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True),
            # AsDiscreted(keys="label", to_onehot=16),
            ToTensord(keys=["image", "label"]),
        ]
    )

    check_ds = Dataset(data=data_dicts, transform=transform)
    check_loader = DataLoader(check_ds, batch_size=1)
    data = first(check_loader)
    # d = AsDiscrete(to_onehot=16)
    print(f"image shape: {data['image'].shape}, label shape: {data['label'].shape}")
    alpha_val = data['label'].clone()
    alpha_val[alpha_val>0] = 0.5

    # img0036 --> slice 140 , img0039 --> 72, img0038 --> 75
    # slice_index = 140
    # plt.figure("blend image and label", (8, 8))
    # plt.title(f"prediction slice {slice_index}")
    # plt.imshow(data["image"][0, 0, :, :, slice_index], cmap="gray")
    # plt.imshow(data["label"][0, 0, :, :, slice_index], alpha=alpha_val[0, 0, :, :, slice_index])
    # plt.show()

    for i in range(1, 10):
        # plot the slice 50 - 100 of image, label and blend result
        slice_index = 5 * i
        plt.figure("blend image and label", (8, 8))
        plt.title(f"prediction slice {slice_index}")
        plt.imshow(data["image"][0, 0, :, :, slice_index], cmap="gray")
        plt.imshow(data["label"][0, 0, :, :, slice_index], alpha=alpha_val[0, 0, :, :, slice_index], )
        plt.show()

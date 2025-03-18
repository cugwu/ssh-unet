import os
from monai import data, transforms
from monai.data import load_decathlon_datalist
import numpy as np


def get_kernels_strides(sizes, spacings):
    """
    Taken from https://github.com/Project-MONAI/tutorials/blob/main/modules/dynunet_pipeline/create_network.py
    """
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def get_loader(args, train_folds=None, val_folds=None):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    if args.mri:
        print("You set MRI mode")
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=args.pos,
                    neg=args.neg,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys="image"),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.Orientationd(keys="image", axcodes="RAS"),
                transforms.SpatialPadd(keys="image", spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode="bilinear"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Spacingd( keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                                     mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=args.pos,
                    neg=args.neg,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear","nearest")),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys="image"),
                transforms.EnsureChannelFirstd(keys="image"),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                transforms.SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Spacingd(keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.CropForegroundd(keys=["image"], source_key="image"),
            ]
        )

    val_post = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys="image", axcodes="RAS"),
            transforms.SpatialPadd(keys="image", spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys="image", a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys="image", source_key="image"),
        ]
    )
    val_post_transforms = transforms.Compose(
        [
            transforms.Invertd(
            keys="pred",
            transform=val_post,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
            transforms.Activationsd(keys="pred", softmax=True),
            transforms.AsDiscreted(keys="pred", argmax=True),
            transforms.KeepLargestConnectedComponentd(keys="pred", num_components=1),
            transforms.AsDiscreted(keys=["pred", "label"], to_onehot=args.out_channels),
        ]
    )

    test_post_transforms = transforms.Compose(
        [
            transforms.Invertd(
                keys="pred",
                transform=test_transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            transforms.Activationsd(keys="pred", softmax=True),
            transforms.AsDiscreted(keys="pred", argmax=True),
            transforms.KeepLargestConnectedComponentd(keys="pred", num_components=1),
            transforms.SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.out_dir, output_ext=".nii.gz",
                                  output_dtype=np.uint8, output_postfix="", separate_folder=False, resample=False, writer="NibabelWriter"),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "test", base_dir=data_dir)
        test_ds =  data.CacheDataset(data=test_files, transform=test_transform, cache_num=args.cache_num, num_workers=args.workers)
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=None,
            pin_memory=True,
        )
        loader = [test_loader, test_post_transforms]
    elif args.val_mode:
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = data.CacheDataset(data=val_files, transform=val_post, cache_num=args.cache_num, num_workers=args.workers)
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=None,
            pin_memory=True
        )
        loader = [val_loader, val_post_transforms]
    else:
        if args.nfolds > 1:
            datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
            kfold = data.create_cross_validation_datalist(datalist, nfolds=args.nfolds, train_folds=train_folds, val_folds=val_folds)
            train_ds = data.CacheDataset(data=kfold['training'], transform=train_transform, cache_num=args.cache_num, num_workers=args.workers)
            train_loader = data.DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                sampler=None,
                pin_memory=True,
            )

            val_ds = data.Dataset(data=kfold['validation'], transform=val_transform)
            val_loader = data.DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=None,
                pin_memory=True,
            )
            loader = [train_loader, val_loader]
            # loader = [val_loader, val_post_transforms]
        else:
            datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
            train_ds = data.CacheDataset(data=datalist, transform=train_transform, cache_num=args.cache_num, cache_rate=1.0, num_workers=args.workers)
            train_loader = data.DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                sampler=None,
                pin_memory=True,
            )
            if args.val_every < args.max_epochs:
                val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
                # val_ds = data.CacheDataset(data=val_files, transform=val_transform, cache_num=round(args.cache_num/5), cache_rate=1.0, num_workers=args.workers)
                val_ds = data.Dataset(data=val_files, transform=val_transform)
                val_loader = data.DataLoader( val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=None, pin_memory=True)
            else: val_loader = None
            loader = [train_loader, val_loader]

    return loader


import numpy as np
from monai import transforms, data
from monai.data import list_data_collate
from .datafold_read import datafold_read
from .custom_transforms import PrepareLabeld


def _train_transforms(roi_size, k_div):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        PrepareLabeld(keys=["label"]),
        transforms.CropForegroundd(
            keys=["image", "label"], source_key="image",
            k_divisible=k_div, allow_smaller=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        # 80% of crops guaranteed to contain tumor — critical for small lesions
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=4,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        # Geometric augmentations
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        transforms.RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
            shear_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="border",
        ),
        transforms.Rand3DElasticd(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.2,
            sigma_range=(5, 8),
            magnitude_range=(50, 150),
            padding_mode="zeros",
        ),
        # Intensity augmentations (heavier ranges + MRI-specific)
        transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=1.0),
        transforms.RandAdjustContrastd(keys="image", gamma=(0.7, 1.5), prob=0.3),
        transforms.RandBiasFieldd(keys="image", coeff_range=(0.0, 0.5), prob=0.3),
        transforms.RandGaussianNoised(keys="image", mean=0.0, std=0.1, prob=0.3),
        transforms.RandGaussianSmoothd(
            keys="image", sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5), prob=0.2,
        ),
    ])


def _val_transforms(roi_size, k_div):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        PrepareLabeld(keys=["label"]),
        transforms.CropForegroundd(
            keys=["image", "label"], source_key="image",
            k_divisible=k_div, allow_smaller=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])


def get_loader(batch_size, data_dir, json_list, fold, roi, min_dims, num_workers):
    roi_size = list(roi)

    train_files, val_files = datafold_read(datalist=json_list, basedir=data_dir, fold=fold, mode="training")
    train_files = [{"image": d["image"], "label": d["label"]} for d in train_files]
    val_files   = [{"image": d["image"], "label": d["label"]} for d in val_files]

    train_ds = data.Dataset(data=train_files, transform=_train_transforms(roi_size, min_dims))
    val_ds   = data.Dataset(data=val_files,   transform=_val_transforms(roi_size, min_dims))

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=list_data_collate,
    )
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader

import json
import os

import torch
from torch.utils.data import ConcatDataset

from monai import transforms, data
from .datafold_read import datafold_read
from .custom_transforms import ConvertToBraTSRegionsd, ConvertToMultiClassd


def _label_transform(scheme):
    if scheme == "regions":
        return ConvertToBraTSRegionsd(keys=["label"])
    if scheme == "multiclass":
        return ConvertToMultiClassd(keys=["label"])
    raise ValueError(f"scheme must be 'regions' or 'multiclass', got {scheme!r}")


def _labeled_transforms(roi_size, k_div, scheme):
    train = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        _label_transform(scheme),
        transforms.CropForegroundd(
            keys=["image", "label"], source_key="image",
            k_divisible=k_div, allow_smaller=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        transforms.RandGaussianNoised(keys="image", mean=0.0, std=0.1, prob=0.15),
        transforms.RandGaussianSmoothd(
            keys="image", sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15), prob=0.15,
        ),
    ])
    val = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        _label_transform(scheme),
        transforms.CropForegroundd(
            keys=["image", "label"], source_key="image",
            k_divisible=k_div, allow_smaller=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])
    return train, val


def _pseudo_transform(roi_size, k_div):
    # Pseudo-labels are saved as a single-channel int16 class map (0=bg, 1=NCR, 2=ED,
    # 3=ET, ignore_index=4 on ambiguous voxels). EnsureChannelFirstd adds the leading
    # channel -> (1, H, W, D); we cast to long WITHOUT ConvertToMultiClassd, which would
    # remap 4 -> 3 and destroy the ignore label. The ignore value is preserved through
    # the nearest-interp spatial augmentations and skipped by IgnoreAwareDiceCELoss.
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        transforms.CastToTyped(keys=["label"], dtype=torch.long),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.CropForegroundd(
            keys=["image", "label"], source_key="image",
            k_divisible=k_div, allow_smaller=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        transforms.RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        transforms.RandGaussianNoised(keys="image", mean=0.0, std=0.1, prob=0.15),
        transforms.RandGaussianSmoothd(
            keys="image", sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15), prob=0.15,
        ),
    ])


def get_loader(
    batch_size,
    data_dir,
    json_list,
    fold,
    roi,
    min_dims,
    num_workers,
    pseudo_image_root=None,
    pseudo_label_root=None,
    scheme="multiclass",
):
    k_div    = min_dims
    roi_size = list(roi)


    train_files, val_files = datafold_read(
        datalist=json_list, basedir=data_dir, fold=fold, mode="training"
    )

    train_files = [{"image": d["image"], "label": d["label"], "is_pseudo": 0} for d in train_files]
    val_files   = [{"image": d["image"], "label": d["label"]} for d in val_files]

    labeled_train_transform, val_transform = _labeled_transforms(roi_size, k_div, scheme)
    labeled_train_ds = data.Dataset(data=train_files, transform=labeled_train_transform)
    val_ds           = data.Dataset(data=val_files,   transform=val_transform)


    with open(json_list) as f:
        full_json = json.load(f)
    pseudo_files = full_json.get("pseudo", [])

    print(
        f"[DataLoader DEBUG] json={json_list}  "
        f"pseudo_entries={len(pseudo_files)}  "
        f"pseudo_image_root={pseudo_image_root!r}  "
        f"pseudo_label_root={pseudo_label_root!r}"
    )

    if pseudo_files and pseudo_image_root and pseudo_label_root and scheme == "regions":
        raise ValueError(
            "pseudo-labels are stored as single-channel class maps with ignore_index=4 "
            "and are only supported by scheme='multiclass' (out_channels: 4). Either "
            "unset pseudo_image_root/pseudo_label_root or switch out_channels to 4."
        )

    if pseudo_files and pseudo_image_root and pseudo_label_root:
        pseudo_files = [
            {
                "image": [os.path.join(pseudo_image_root, p) for p in e["image"]],
                "label": os.path.join(pseudo_label_root, e["label"]),
                "is_pseudo": 1,
            }
            for e in pseudo_files
        ]
        pseudo_ds = data.Dataset(data=pseudo_files, transform=_pseudo_transform(roi_size, k_div))
        train_ds  = ConcatDataset([labeled_train_ds, pseudo_ds])
    else:
        pseudo_ds = None
        train_ds  = labeled_train_ds

    print(
        f"[DataLoader] fold={fold}  GT-train={len(train_files)}  "
        f"pseudo-train={len(pseudo_files) if pseudo_ds else 0}  GT-val={len(val_files)}"
    )

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
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

"""
Evaluate SmallTumorHRSeg on all 24 dataset examples and report per-case
and aggregate Dice scores for BG / NCR / ED / ET.

Usage:
    python eval_all.py --checkpoint /path/to/model.pth
    python eval_all.py  # uses default checkpoint path from train.yaml
"""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from monai import data, transforms
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from script import SmallTumorHRSeg
from scripts.utils.custom_transforms import PrepareLabeld


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_cfg: dict, device: torch.device) -> torch.nn.Module:
    cfg = model_cfg.get("model", {})
    return SmallTumorHRSeg(
        in_channels=int(cfg.get("in_channels", 4)),
        num_classes=int(cfg.get("num_classes", 4)),
        h_ch=int(cfg.get("h_ch", 32)),
        m_ch=int(cfg.get("m_ch", 64)),
        num_stages=int(cfg.get("num_stages", 3)),
        blocks_per_stage=int(cfg.get("blocks_per_stage", 2)),
    ).to(device)


def load_all_cases(json_list: str, data_dir: str) -> list[dict]:
    with open(json_list) as f:
        raw = json.load(f)
    entries = raw["training"]
    files = []
    for d in entries:
        img = d["image"]
        lbl = d["label"]
        if isinstance(img, list):
            img = [os.path.join(data_dir, p) for p in img]
        else:
            img = os.path.join(data_dir, img)
        if isinstance(lbl, list):
            lbl = [os.path.join(data_dir, p) for p in lbl]
        else:
            lbl = os.path.join(data_dir, lbl)
        files.append({"image": img, "label": lbl})
    return files


def val_transforms(roi_size, min_dims):
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        PrepareLabeld(keys=["label"]),
        transforms.CropForegroundd(
            keys=["image", "label"], source_key="image",
            k_divisible=min_dims, allow_smaller=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])


def main():
    parser = argparse.ArgumentParser(description="Evaluate HRNet on all 24 cases")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()

    train_cfg = load_yaml(REPO_ROOT / args.train_config)
    model_cfg = load_yaml(REPO_ROOT / args.model_config)

    data_cfg = train_cfg["data"]
    data_dir = data_cfg.get("root_dir", "./data")
    json_list = data_cfg.get("json_list", "./data/dataset.json")
    roi = tuple(data_cfg.get("roi", [128, 128, 128]))
    sw_batch_size = int(data_cfg.get("sw_batch_size", 4))
    infer_overlap = float(data_cfg.get("infer_overlap", 0.5))
    num_workers = int(data_cfg.get("num_workers", 4))
    min_dims = [32, 32, 32]

    checkpoint_path = args.checkpoint or train_cfg.get("checkpoint", {}).get("resume")
    if checkpoint_path is None:
        # try the one discovered in the repo
        default_ckpt = REPO_ROOT.parent / "ckpt" / "model_best_fold_1.pth"
        if default_ckpt.exists():
            checkpoint_path = str(default_ckpt)
        else:
            parser.error(
                "No checkpoint found. Pass --checkpoint /path/to/model.pth"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Data dir        : {data_dir}")
    print(f"ROI             : {roi}")
    print(f"Infer overlap   : {infer_overlap}")


    model = build_model(model_cfg, device)
    num_classes = int(model_cfg.get("model", {}).get("num_classes", 4))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint best : {ckpt.get('best_acc', 'unknown')}")


    all_files = load_all_cases(json_list, data_dir)
    print(f"\nTotal cases     : {len(all_files)}")

    ds = data.Dataset(
        data=all_files,
        transform=val_transforms(list(roi), min_dims),
    )
    loader = data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


    model_inferer = partial(
        sliding_window_inference,
        roi_size=list(roi),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )
    post_softmax = Activations(softmax=True)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    post_label = AsDiscrete(to_onehot=num_classes)

    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )


    rows = []
    print(f"\n{'Case':<6}  {'BG':>8}  {'NCR':>8}  {'ED':>8}  {'ET':>8}  {'Mean(fg)':>8}")
    print("-" * 58)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model_inferer(images)

            preds = [post_pred(post_softmax(t)) for t in decollate_batch(logits)]
            gts = [post_label(t) for t in decollate_batch(labels)]

            dice_metric.reset()
            dice_metric(y_pred=preds, y=gts)
            acc, not_nans = dice_metric.aggregate()
            bg, ncr, ed, et = acc[0].item(), acc[1].item(), acc[2].item(), acc[3].item()
            mean = np.nanmean([ncr, ed, et])

            rows.append((idx + 1, bg, ncr, ed, et, mean))
            print(f"{idx+1:<6}  {bg:>8.4f}  {ncr:>8.4f}  {ed:>8.4f}  {et:>8.4f}  {mean:>8.4f}")


    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows])
    print("-" * 58)
    print(f"{'Mean':<6}  {arr[:,0].mean():>8.4f}  {arr[:,1].mean():>8.4f}  {arr[:,2].mean():>8.4f}  {arr[:,3].mean():>8.4f}  {arr[:,4].mean():>8.4f}")
    print(f"{'Std':<6}  {arr[:,0].std():>8.4f}  {arr[:,1].std():>8.4f}  {arr[:,2].std():>8.4f}  {arr[:,3].std():>8.4f}  {arr[:,4].std():>8.4f}")
    print(f"{'Min':<6}  {arr[:,0].min():>8.4f}  {arr[:,1].min():>8.4f}  {arr[:,2].min():>8.4f}  {arr[:,3].min():>8.4f}  {arr[:,4].min():>8.4f}")
    print(f"{'Max':<6}  {arr[:,0].max():>8.4f}  {arr[:,1].max():>8.4f}  {arr[:,2].max():>8.4f}  {arr[:,3].max():>8.4f}  {arr[:,4].max():>8.4f}")
    print(f"\nOverall mean foreground Dice (NCR/ED/ET) across all cases: {arr[:,4].mean():.4f}")


if __name__ == "__main__":
    main()

# STANDARD LIBRARY
import argparse
import csv
import datetime
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch

# MONAI IMPORTS
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

# CUSTOM IMPORTS
from scripts.utils.AverageMeter import AverageMeter
from scripts.utils.datafold_read import datafold_read
from scripts.utils.dataloader import get_loader
from src.main import REPO_ROOT, build_model, load_yaml, resolve_path


REGION_NAMES = ["ET", "TC", "WT"]


def labels_to_regions(seg: torch.Tensor) -> torch.Tensor:
    """Convert a mutually-exclusive class map (0=bg, 1=NCR, 2=ED, 3=ET) into the
    overlapping BraTS region masks, stacked as [ET, TC, WT]:
        ET: label 3
        TC: labels 1, 3
        WT: labels 1, 2, 3

    Accepts (1, H, W, D) or (H, W, D); returns (3, H, W, D) float tensor.
    """
    if seg.dim() == 4:
        seg = seg[0]
    et = seg == 3
    tc = (seg == 1) | (seg == 3)
    wt = (seg == 1) | (seg == 2) | (seg == 3)
    return torch.stack([et, tc, wt], dim=0).float()


def case_id_from_entry(entry):
    image_path = entry["image"][0] if isinstance(entry["image"], list) else entry["image"]
    return Path(image_path).parent.name or Path(image_path).stem.split(".")[0]


def run_validation(
    train_config_path: str,
    model_config_path: str,
    checkpoint_path: str,
    data_dir: str | None = None,
    json_list: str | None = None,
    fold: int | None = None,
    output_dir: str | None = None,
    infer_overlap: float | None = None,
    sw_batch_size: int | None = None,
    num_workers: int | None = None,
):
    train_config = load_yaml(resolve_path(train_config_path))
    model_config = load_yaml(resolve_path(model_config_path))

    data_cfg = train_config.get("data", {})

    roi = list(data_cfg.get("roi", [128, 128, 128]))
    min_dims = [32, 32, 32]

    data_dir = resolve_path(data_dir) if data_dir else resolve_path(data_cfg.get("root_dir", "./data"))
    json_list = resolve_path(json_list) if json_list else resolve_path(data_cfg.get("json_list", "./data/datalist.example.json"))
    fold = int(data_cfg.get("fold", 0) if fold is None else fold)
    infer_overlap = float(data_cfg.get("infer_overlap", 0.5) if infer_overlap is None else infer_overlap)
    sw_batch_size = int(data_cfg.get("sw_batch_size", 1) if sw_batch_size is None else sw_batch_size)
    num_workers = int(data_cfg.get("num_workers", 4) if num_workers is None else num_workers)

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        output_dir = os.path.join(REPO_ROOT, "outputs", "validation", timestamp)
    else:
        output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = resolve_path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    # Case ids in the exact order get_loader's val split will yield them
    # (datafold_read is deterministic given the same json_list/fold).
    _, val_entries = datafold_read(datalist=json_list, basedir=data_dir, fold=fold, mode="training")
    case_ids = [case_id_from_entry(e) for e in val_entries]

    # batch_size only affects the train loader; get_loader always builds the
    # val loader with batch_size=1.
    _, val_loader = get_loader(
        batch_size=1,
        data_dir=data_dir,
        json_list=json_list,
        fold=fold,
        roi=roi,
        min_dims=min_dims,
        num_workers=num_workers,
    )

    if len(val_loader) != len(case_ids):
        raise RuntimeError(
            f"Val loader length ({len(val_loader)}) does not match case id count "
            f"({len(case_ids)}); cannot align per-case CSV rows."
        )

    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    running = AverageMeter()
    per_case_rows = []

    with torch.no_grad():
        for case_id, batch_data in zip(case_ids, val_loader):
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)

            logits = model_inferer(image).float()
            pred_labels = torch.argmax(logits, dim=1, keepdim=True)  # (1, 1, H, W, D)

            pred_regions = [labels_to_regions(p) for p in decollate_batch(pred_labels)]
            gt_regions = [labels_to_regions(g) for g in decollate_batch(label)]

            dice_metric.reset()
            dice_metric(y_pred=pred_regions, y=gt_regions)
            case_dice, case_not_nans = dice_metric.aggregate()
            case_dice = case_dice.cpu().numpy()
            running.update(case_dice, n=case_not_nans.cpu().numpy())

            row = {"case_id": case_id}
            for name, value in zip(REGION_NAMES, case_dice):
                row[f"dice_{name}"] = float(value)
            per_case_rows.append(row)

            print(f"{case_id}: ET={case_dice[0]:.4f}  TC={case_dice[1]:.4f}  WT={case_dice[2]:.4f}")

    csv_path = os.path.join(output_dir, "region_dice.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "dice_ET", "dice_TC", "dice_WT"])
        writer.writeheader()
        writer.writerows(per_case_rows)

    mean_et, mean_tc, mean_wt = running.avg
    mean_overall = float(np.mean(running.avg))

    print("\n==== Validation Region Dice Summary ====")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Fold: {fold}   Cases: {len(case_ids)}")
    print(f"Dice_ET: {mean_et:.4f}")
    print(f"Dice_TC: {mean_tc:.4f}")
    print(f"Dice_WT: {mean_wt:.4f}")
    print(f"Dice_Mean(ET/TC/WT): {mean_overall:.4f}")
    print(f"Per-case CSV written to: {csv_path}")

    return {"ET": mean_et, "TC": mean_tc, "WT": mean_wt, "mean": mean_overall, "csv_path": csv_path}


def cli():
    parser = argparse.ArgumentParser(description="Report ET/TC/WT validation dice for a trained SwinUNETR checkpoint.")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Path to the training config YAML.")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to the model config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--data-dir", default=None, help="Override the labeled data root directory.")
    parser.add_argument("--json-list", default=None, help="Override the datalist JSON.")
    parser.add_argument("--fold", type=int, default=None, help="Fold to use for the train/val split.")
    parser.add_argument("--output-dir", default=None, help="Directory to write the per-case CSV. Auto-generated under outputs/validation/<timestamp> if omitted.")
    parser.add_argument("--infer-overlap", type=float, default=None, help="Sliding window inference overlap override.")
    parser.add_argument("--sw-batch-size", type=int, default=None, help="Sliding window batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader worker count override.")

    args = parser.parse_args()

    run_validation(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        json_list=args.json_list,
        fold=args.fold,
        output_dir=args.output_dir,
        infer_overlap=args.infer_overlap,
        sw_batch_size=args.sw_batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    cli()

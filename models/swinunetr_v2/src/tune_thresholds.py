# STANDARD LIBRARY
import argparse
import csv
import datetime
import json
import os
from functools import partial

import numpy as np
import torch

# MONAI IMPORTS
from monai.inferers import sliding_window_inference

# CUSTOM IMPORTS
from scripts.utils.datafold_read import datafold_read
from scripts.utils.dataloader import get_loader
from src.main import REPO_ROOT, build_model, load_yaml, resolve_path
from src.predict import REGION_NAMES, resolve_scheme_and_model_config
from src.validate import case_id_from_entry, labels_to_regions
from src.validate_folds import resolve_fold_checkpoints


def build_threshold_grid(t_min: float, t_max: float, t_step: float) -> np.ndarray:
    n_steps = int(round((t_max - t_min) / t_step)) + 1
    return np.round(t_min + t_step * np.arange(n_steps), 6)


def sweep_region(prob: torch.Tensor, gt_bool: torch.Tensor, thresholds: np.ndarray) -> np.ndarray:


    gt_sum = int(gt_bool.sum().item())
    out = np.empty(len(thresholds), dtype=np.float64)
    for i, t in enumerate(thresholds):
        pred_bool = prob > t
        inter = int((pred_bool & gt_bool).sum().item())
        pred_sum = int(pred_bool.sum().item())
        if gt_sum == 0 and pred_sum == 0:
            out[i] = np.nan
        elif gt_sum == 0:
            out[i] = 0.0
        else:
            fp = pred_sum - inter
            fn = gt_sum - inter
            out[i] = 2.0 * inter / (2 * inter + fp + fn)
    return out


def run_fold_sweep(
    model_config: dict,
    checkpoint_path: str,
    fold: int,
    data_dir: str,
    json_list: str,
    roi: list,
    min_dims: list,
    infer_overlap: float,
    sw_batch_size: int,
    num_workers: int,
    thresholds: np.ndarray,
    device: torch.device,
):
    
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


    _, val_entries = datafold_read(datalist=json_list, basedir=data_dir, fold=fold, mode="training")
    case_ids = [case_id_from_entry(e) for e in val_entries]

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
            f"({len(case_ids)}) for fold {fold}."
        )

    sums = {r: np.zeros(len(thresholds)) for r in REGION_NAMES}
    counts = {r: np.zeros(len(thresholds)) for r in REGION_NAMES}

    with torch.no_grad():
        for case_id, batch_data in zip(case_ids, val_loader):
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)

            logits = model_inferer(image).float()
            probs = torch.sigmoid(logits)[0]  # (3, H, W, D), channel order ET/TC/WT

            gt = labels_to_regions(label[0])  # (3, H, W, D) -> ET, TC, WT bool-as-float


            region_probs = {"ET": probs[0], "TC": probs[1], "WT": probs[2]}
            region_gts = {"ET": gt[0].bool(), "TC": gt[1].bool(), "WT": gt[2].bool()}

            for r in REGION_NAMES:
                dices = sweep_region(region_probs[r], region_gts[r], thresholds)
                valid = ~np.isnan(dices)
                sums[r][valid] += dices[valid]
                counts[r][valid] += 1

            print(f"  fold {fold} | {case_id}: swept {len(thresholds)} thresholds/region")

    return sums, counts, len(case_ids)


def tune_thresholds(
    train_config_path: str,
    model_config_path: str,
    checkpoint_dir: str | None = None,
    checkpoints: dict[int, str] | None = None,
    folds: list[int] | None = None,
    pattern: str = "model_best_fold_{fold}.pth",
    data_dir: str | None = None,
    json_list: str | None = None,
    output_dir: str | None = None,
    infer_overlap: float | None = None,
    sw_batch_size: int | None = None,
    num_workers: int | None = None,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.05,
):
    
    train_config = load_yaml(resolve_path(train_config_path))
    model_config = load_yaml(resolve_path(model_config_path))

    data_cfg = train_config.get("data", {})
    roi = list(data_cfg.get("roi", [128, 128, 128]))
    min_dims = [32, 32, 32]

    _, _, model_config = resolve_scheme_and_model_config(model_config, "regions")

    data_dir = resolve_path(data_dir) if data_dir else resolve_path(data_cfg.get("root_dir", "./data"))
    json_list = resolve_path(json_list) if json_list else resolve_path(data_cfg.get("json_list", "./data/datalist.example.json"))
    infer_overlap = float(data_cfg.get("infer_overlap", 0.5) if infer_overlap is None else infer_overlap)
    sw_batch_size = int(data_cfg.get("sw_batch_size", 1) if sw_batch_size is None else sw_batch_size)
    num_workers = int(data_cfg.get("num_workers", 4) if num_workers is None else num_workers)

    folds = folds if folds is not None else [0, 1, 2, 3, 4]
    if checkpoints is None:
        checkpoints = resolve_fold_checkpoints(checkpoint_dir or "./checkpoints", folds, pattern)
    else:
        missing = [f for f in folds if f not in checkpoints]
        if missing:
            raise ValueError(f"--checkpoint was not given for fold(s): {missing}")

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        output_dir = os.path.join(REPO_ROOT, "outputs", "threshold_sweep", timestamp)
    else:
        output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = build_threshold_grid(threshold_min, threshold_max, threshold_step)

    print(f"Threshold grid: {thresholds.tolist()}")
    print(f"Sweeping {len(folds)} fold(s) (pooled OOF): {folds}")
    for fold in folds:
        print(f"  fold {fold}: {checkpoints[fold]}")

    pooled_sums = {r: np.zeros(len(thresholds)) for r in REGION_NAMES}
    pooled_counts = {r: np.zeros(len(thresholds)) for r in REGION_NAMES}
    total_cases = 0

    for fold in folds:
        print(f"\n==== Fold {fold} (OOF) ====")
        sums, counts, n_cases = run_fold_sweep(
            model_config=model_config,
            checkpoint_path=checkpoints[fold],
            fold=fold,
            data_dir=data_dir,
            json_list=json_list,
            roi=roi,
            min_dims=min_dims,
            infer_overlap=infer_overlap,
            sw_batch_size=sw_batch_size,
            num_workers=num_workers,
            thresholds=thresholds,
            device=device,
        )
        total_cases += n_cases
        for r in REGION_NAMES:
            pooled_sums[r] += sums[r]
            pooled_counts[r] += counts[r]

    mean_dice = {r: np.divide(pooled_sums[r], pooled_counts[r], out=np.full(len(thresholds), np.nan), where=pooled_counts[r] > 0) for r in REGION_NAMES}

    best_idx = {r: int(np.nanargmax(mean_dice[r])) for r in REGION_NAMES}
    best_threshold = {r: float(thresholds[best_idx[r]]) for r in REGION_NAMES}
    best_dice = {r: float(mean_dice[r][best_idx[r]]) for r in REGION_NAMES}

    baseline_idx = {r: int(np.argmin(np.abs(thresholds - 0.5))) for r in REGION_NAMES}
    baseline_dice = {r: float(mean_dice[r][baseline_idx[r]]) for r in REGION_NAMES}

    sweep_csv_path = os.path.join(output_dir, "threshold_sweep.csv")
    with open(sweep_csv_path, "w", newline="") as f:
        fieldnames = ["threshold"] + [f"dice_{r}" for r in REGION_NAMES] + [f"n_{r}" for r in REGION_NAMES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(thresholds):
            row = {"threshold": float(t)}
            for r in REGION_NAMES:
                row[f"dice_{r}"] = float(mean_dice[r][i])
                row[f"n_{r}"] = int(pooled_counts[r][i])
            writer.writerow(row)

    best_path = os.path.join(output_dir, "best_thresholds.json")
    with open(best_path, "w") as f:
        json.dump(
            {
                "threshold": best_threshold,
                "dice_at_best": best_dice,
                "dice_at_0.5": baseline_dice,
                "total_oof_cases": total_cases,
                "folds": folds,
            },
            f,
            indent=2,
        )

    print("\n==== Per-Region Best Threshold (pooled OOF) ====")
    header = f"{'Region':<8}{'Best thr':>10}{'Dice@best':>12}{'Dice@0.5':>12}{'Delta':>10}"
    print(header)
    for r in REGION_NAMES:
        delta = best_dice[r] - baseline_dice[r]
        print(f"{r:<8}{best_threshold[r]:>10.3f}{best_dice[r]:>12.4f}{baseline_dice[r]:>12.4f}{delta:>+10.4f}")
    print(f"\nOOF cases pooled: {total_cases}")
    print(f"Full sweep CSV: {sweep_csv_path}")
    print(f"Best thresholds JSON: {best_path}")
    print(
        "\nUse with predict.py/ensemble_predict.py: "
        f"--threshold-et {best_threshold['ET']:.3f} --threshold-tc {best_threshold['TC']:.3f} --threshold-wt {best_threshold['WT']:.3f}"
    )

    return {
        "threshold": best_threshold,
        "dice_at_best": best_dice,
        "dice_at_0.5": baseline_dice,
        "sweep_csv_path": sweep_csv_path,
        "best_thresholds_path": best_path,
    }


def cli():
    parser = argparse.ArgumentParser(
        description="Sweep a per-region binary threshold (ET/TC/WT) on pooled out-of-fold predictions "
        "from the region-teacher (out_channels=3, sigmoid) model, maximizing Dice per region."
    )
    parser.add_argument("--train-config", default="configs/train.yaml", help="Path to the training config YAML.")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to the model config YAML.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory containing one checkpoint per fold (default: ./checkpoints).")
    parser.add_argument(
        "--pattern",
        default="model_best_fold_{fold}.pth",
        help="Filename template for fold checkpoints inside --checkpoint-dir, must contain '{fold}'.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        default=None,
        metavar="FOLD=PATH",
        help="Explicit per-fold checkpoint path, e.g. --checkpoint 0=/path/a.pth. Overrides --checkpoint-dir/--pattern. Repeat once per fold.",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Folds to evaluate. Default: 0 1 2 3 4.")
    parser.add_argument("--data-dir", default=None, help="Override the labeled data root directory.")
    parser.add_argument("--json-list", default=None, help="Override the datalist JSON.")
    parser.add_argument("--output-dir", default=None, help="Directory to write the sweep CSV and best-thresholds JSON. Auto-generated under outputs/threshold_sweep/<timestamp> if omitted.")
    parser.add_argument("--infer-overlap", type=float, default=None, help="Sliding window inference overlap override.")
    parser.add_argument("--sw-batch-size", type=int, default=None, help="Sliding window batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader worker count override.")
    parser.add_argument("--threshold-min", type=float, default=0.05, help="Lowest threshold in the sweep grid.")
    parser.add_argument("--threshold-max", type=float, default=0.95, help="Highest threshold in the sweep grid.")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Step size of the sweep grid.")

    args = parser.parse_args()

    if "{fold}" not in args.pattern:
        parser.error("--pattern must contain the literal '{fold}' placeholder")

    checkpoints = None
    if args.checkpoints:
        checkpoints = {}
        for item in args.checkpoints:
            if "=" not in item:
                parser.error(f"--checkpoint must be FOLD=PATH, got {item!r}")
            fold_str, path = item.split("=", 1)
            checkpoints[int(fold_str)] = path

    tune_thresholds(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        checkpoint_dir=args.checkpoint_dir,
        checkpoints=checkpoints,
        folds=args.folds,
        pattern=args.pattern,
        data_dir=args.data_dir,
        json_list=args.json_list,
        output_dir=args.output_dir,
        infer_overlap=args.infer_overlap,
        sw_batch_size=args.sw_batch_size,
        num_workers=args.num_workers,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )


if __name__ == "__main__":
    cli()

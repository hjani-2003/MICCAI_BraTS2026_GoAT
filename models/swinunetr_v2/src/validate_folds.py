# STANDARD LIBRARY
import argparse
import csv
import datetime
import os
from pathlib import Path

import numpy as np

# CUSTOM IMPORTS
from src.main import REPO_ROOT, resolve_path
from src.validate import run_validation


def resolve_fold_checkpoints(checkpoint_dir: str, folds: list[int], pattern: str) -> dict[int, str]:
    
    checkpoint_dir = Path(resolve_path(checkpoint_dir))
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"checkpoint_dir does not exist: {checkpoint_dir}")

    resolved = {}
    missing = []
    for fold in folds:
        candidate = checkpoint_dir / pattern.format(fold=fold)
        if candidate.is_file():
            resolved[fold] = str(candidate)
            continue
        matches = sorted(checkpoint_dir.glob(pattern.format(fold=fold).replace(".pth", "*").replace(".pt", "*")))
        matches = [m for m in matches if m.suffix in (".pth", ".pt")]
        if len(matches) == 1:
            resolved[fold] = str(matches[0])
        elif len(matches) > 1:
            raise FileNotFoundError(
                f"Fold {fold}: multiple checkpoints matched pattern {pattern.format(fold=fold)!r} in "
                f"{checkpoint_dir}: {[str(m) for m in matches]}. Narrow --pattern or rename files."
            )
        else:
            missing.append(fold)

    if missing:
        available = sorted(p.name for p in checkpoint_dir.glob("*.pth")) + sorted(p.name for p in checkpoint_dir.glob("*.pt"))
        raise FileNotFoundError(
            f"No checkpoint found for fold(s) {missing} in {checkpoint_dir} using pattern {pattern!r}. "
            f"Files present: {available}"
        )
    return resolved


def run_cross_validation(
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
):
    """Run per-fold validation: for each fold, load that fold's checkpoint and
    evaluate on that fold's held-out validation split (as determined by
    datafold_read), reporting ET/TC/WT Dice. Aggregates a cross-validation
    summary across all folds."""
    folds = folds if folds is not None else [0, 1, 2, 3, 4]

    if checkpoints is None:
        checkpoints = resolve_fold_checkpoints(checkpoint_dir or "./checkpoints", folds, pattern)
    else:
        missing = [f for f in folds if f not in checkpoints]
        if missing:
            raise ValueError(f"--checkpoint was not given for fold(s): {missing}")

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        output_dir = os.path.join(REPO_ROOT, "outputs", "cv_validation", timestamp)
    else:
        output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Cross-validation over {len(folds)} fold(s): {folds}")
    for fold in folds:
        print(f"  fold {fold}: {checkpoints[fold]}")

    per_fold_rows = []
    for fold in folds:
        print(f"\n==== Fold {fold} ====")
        result = run_validation(
            train_config_path=train_config_path,
            model_config_path=model_config_path,
            checkpoint_path=checkpoints[fold],
            data_dir=data_dir,
            json_list=json_list,
            fold=fold,
            output_dir=os.path.join(output_dir, f"fold_{fold}"),
            infer_overlap=infer_overlap,
            sw_batch_size=sw_batch_size,
            num_workers=num_workers,
        )
        per_fold_rows.append(
            {
                "fold": fold,
                "checkpoint": checkpoints[fold],
                "dice_ET": result["ET"],
                "dice_TC": result["TC"],
                "dice_WT": result["WT"],
                "dice_mean": result["mean"],
            }
        )

    et = np.array([r["dice_ET"] for r in per_fold_rows])
    tc = np.array([r["dice_TC"] for r in per_fold_rows])
    wt = np.array([r["dice_WT"] for r in per_fold_rows])
    overall_mean = np.array([r["dice_mean"] for r in per_fold_rows])

    summary_path = os.path.join(output_dir, "cv_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "checkpoint", "dice_ET", "dice_TC", "dice_WT", "dice_mean"])
        writer.writeheader()
        writer.writerows(per_fold_rows)
        writer.writerow(
            {
                "fold": "mean",
                "checkpoint": "",
                "dice_ET": float(et.mean()),
                "dice_TC": float(tc.mean()),
                "dice_WT": float(wt.mean()),
                "dice_mean": float(overall_mean.mean()),
            }
        )
        writer.writerow(
            {
                "fold": "std",
                "checkpoint": "",
                "dice_ET": float(et.std()),
                "dice_TC": float(tc.std()),
                "dice_WT": float(wt.std()),
                "dice_mean": float(overall_mean.std()),
            }
        )

    print("\n==== Cross-Validation Summary ====")
    header = f"{'Fold':<6}{'ET':>8}{'TC':>8}{'WT':>8}{'Mean':>8}"
    print(header)
    for r in per_fold_rows:
        print(f"{r['fold']:<6}{r['dice_ET']:>8.4f}{r['dice_TC']:>8.4f}{r['dice_WT']:>8.4f}{r['dice_mean']:>8.4f}")
    print("-" * len(header))
    print(f"{'mean':<6}{et.mean():>8.4f}{tc.mean():>8.4f}{wt.mean():>8.4f}{overall_mean.mean():>8.4f}")
    print(f"{'std':<6}{et.std():>8.4f}{tc.std():>8.4f}{wt.std():>8.4f}{overall_mean.std():>8.4f}")
    print(f"\nSummary CSV written to: {summary_path}")

    return {
        "per_fold": per_fold_rows,
        "mean": {"ET": float(et.mean()), "TC": float(tc.mean()), "WT": float(wt.mean()), "overall": float(overall_mean.mean())},
        "std": {"ET": float(et.std()), "TC": float(tc.std()), "WT": float(wt.std()), "overall": float(overall_mean.std())},
        "csv_path": summary_path,
    }


def _parse_checkpoint_arg(values: list[str] | None) -> dict[int, str] | None:
    if not values:
        return None
    result = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--checkpoint must be FOLD=PATH, got {item!r}")
        fold_str, path = item.split("=", 1)
        result[int(fold_str)] = path
    return result


def cli():
    parser = argparse.ArgumentParser(
        description="Run per-fold cross-validation: load each fold's checkpoint, evaluate on that fold's "
        "held-out split, and report ET/TC/WT Dice per fold plus a cross-validation summary."
    )
    parser.add_argument("--train-config", default="configs/train.yaml", help="Path to the training config YAML.")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to the model config YAML.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory containing one checkpoint per fold (default: ./checkpoints).")
    parser.add_argument(
        "--pattern",
        default="model_best_fold_{fold}.pth",
        help="Filename template for fold checkpoints inside --checkpoint-dir, must contain '{fold}'. "
        "Default matches src/main.py's save naming: 'model_best_fold_{fold}.pth'.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        default=None,
        metavar="FOLD=PATH",
        help="Explicit per-fold checkpoint path, e.g. --checkpoint 0=/path/a.pth --checkpoint 1=/path/b.pth. "
        "Overrides --checkpoint-dir/--pattern discovery. Repeat once per fold.",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Folds to evaluate. Default: 0 1 2 3 4.")
    parser.add_argument("--data-dir", default=None, help="Override the labeled data root directory.")
    parser.add_argument("--json-list", default=None, help="Override the datalist JSON.")
    parser.add_argument("--output-dir", default=None, help="Directory to write per-fold and summary CSVs. Auto-generated under outputs/cv_validation/<timestamp> if omitted.")
    parser.add_argument("--infer-overlap", type=float, default=None, help="Sliding window inference overlap override.")
    parser.add_argument("--sw-batch-size", type=int, default=None, help="Sliding window batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader worker count override.")

    args = parser.parse_args()

    if "{fold}" not in args.pattern:
        parser.error("--pattern must contain the literal '{fold}' placeholder")

    run_cross_validation(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        checkpoint_dir=args.checkpoint_dir,
        checkpoints=_parse_checkpoint_arg(args.checkpoints),
        folds=args.folds,
        pattern=args.pattern,
        data_dir=args.data_dir,
        json_list=args.json_list,
        output_dir=args.output_dir,
        infer_overlap=args.infer_overlap,
        sw_batch_size=args.sw_batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    cli()

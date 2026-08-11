"""
Compute ET / TC / WT Dice scores from nnUNet predictions vs BraTS ground truth.

The nnUNet output folder contains integer-label NIfTI files (0=bg, 1=NCR, 2=SNFH, 3=ET).
We derive the three BraTS regions:
  ET  = {3}
  TC  = {1, 3}
  WT  = {1, 2, 3}

Usage
-----
python scripts/evaluate.py \
    --pred-dir   /path/to/nnunet_predictions \
    --label-dir  /path/to/nnUNet_raw/Dataset001_BraTS/labelsTr \
    [--fold-json /path/to/folds.json --fold 0]
    [--output    results.csv]
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


# Region definitions

REGIONS = {
    "ET": {3},
    "TC": {1, 3},
    "WT": {1, 2, 3},
}


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0  # both empty -> perfect score
    return float(2.0 * intersection / denom)


def load_vol(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj, dtype=np.int16)


def evaluate_case(pred_path: str, gt_path: str) -> dict[str, float]:
    pred = load_vol(pred_path)
    gt   = load_vol(gt_path)

    scores = {}
    for region, labels in REGIONS.items():
        pred_mask = np.isin(pred, list(labels))
        gt_mask   = np.isin(gt,   list(labels))
        scores[region] = dice(pred_mask, gt_mask)
    return scores


def find_cases(pred_dir: Path, label_dir: Path, fold_json: str | None, fold: int | None):
    """
    Yield (pred_path, gt_path, case_id) tuples.

    If fold_json + fold are provided, restrict to the validation set of that fold.
    Otherwise evaluate every NIfTI in pred_dir.
    """
    if fold_json and fold is not None:
        with open(fold_json) as f:
            raw = json.load(f)
        val_cases = set()
        for item in raw["training"]:
            if int(item.get("fold", -1)) == fold:
                case_id = Path(item["image"][0]).parent.name
                if not case_id or case_id == ".":
                    case_id = Path(item["image"][0].split("/")[0]).name
                val_cases.add(case_id)
    else:
        val_cases = None

    for pred_file in sorted(pred_dir.glob("*.nii.gz")):
        case_id = pred_file.stem.replace(".nii", "")  # strip .nii.gz
        if val_cases is not None and case_id not in val_cases:
            continue
        gt_file = label_dir / f"{case_id}.nii.gz"
        if not gt_file.exists():
            print(f"[WARN] GT not found for {case_id}, skipping")
            continue
        yield str(pred_file), str(gt_file), case_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir",  required=True, help="Folder with nnUNet predictions.")
    parser.add_argument("--label-dir", required=True, help="Folder with ground-truth labels.")
    parser.add_argument("--fold-json", default=None,  help="Optional: restrict to one fold's val set.")
    parser.add_argument("--fold",      type=int, default=None)
    parser.add_argument("--output",    default=None,  help="Optional CSV output path.")
    args = parser.parse_args()

    pred_dir  = Path(args.pred_dir)
    label_dir = Path(args.label_dir)

    rows = []
    for pred_path, gt_path, case_id in find_cases(pred_dir, label_dir, args.fold_json, args.fold):
        scores = evaluate_case(pred_path, gt_path)
        rows.append({"case": case_id, **scores})
        print(f"{case_id:40s}  ET={scores['ET']:.4f}  TC={scores['TC']:.4f}  WT={scores['WT']:.4f}")

    if not rows:
        print("No cases evaluated.")
        return

    et_scores = [r["ET"] for r in rows]
    tc_scores = [r["TC"] for r in rows]
    wt_scores = [r["WT"] for r in rows]
    avg = (np.mean(et_scores) + np.mean(tc_scores) + np.mean(wt_scores)) / 3

    print(f"\n{'─'*60}")
    print(f"Cases evaluated : {len(rows)}")
    print(f"ET Dice         : {np.mean(et_scores):.4f} ± {np.std(et_scores):.4f}")
    print(f"TC Dice         : {np.mean(tc_scores):.4f} ± {np.std(tc_scores):.4f}")
    print(f"WT Dice         : {np.mean(wt_scores):.4f} ± {np.std(wt_scores):.4f}")
    print(f"Mean Dice       : {avg:.4f}")

    if args.output:
        fieldnames = ["case", "ET", "TC", "WT"]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-case results saved to {args.output}")


if __name__ == "__main__":
    main()

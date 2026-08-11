"""
Convert the BraTS fold-JSON into nnUNet's splits_final.json so that
nnUNet trains on exactly the same fold splits as your other models.

nnUNet v2 expects splits_final.json at:
  {nnUNet_preprocessed}/Dataset{id:03d}_{name}/splits_final.json

Format:
  [
    {"train": ["case_id_1", ...], "val": ["case_id_7", ...]},
    ...   (one dict per fold)
  ]

Run AFTER `nnUNetv2_plan_and_preprocess` (which creates the preprocessed dir).

Usage
-----
python data/generate_nnunet_splits.py --config configs/paths.yaml
"""

import argparse
import json
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_folds(json_list: str, data_dir: str) -> dict[int, dict]:
    """Return {fold_id: {"train": [case_ids], "val": [case_ids]}}."""
    with open(json_list) as f:
        raw = json.load(f)

    all_folds: dict[int, list[str]] = {}
    for item in raw["training"]:
        fold = int(item.get("fold", 0))
        images = item["image"]
        # Derive case_id the same way as prepare_nnunet_dataset.py
        case_id = Path(images[0] if not Path(images[0]).is_absolute()
                       else images[0]).parent.name
        if not case_id or case_id == ".":
            # Fallback: strip directory from the relative path first segment
            case_id = Path(images[0].split("/")[0]).name

        all_folds.setdefault(fold, []).append(case_id)

    n_folds = max(all_folds.keys()) + 1
    all_cases = [c for cases in all_folds.values() for c in cases]

    splits = []
    for fold_id in range(n_folds):
        val_cases   = all_folds.get(fold_id, [])
        train_cases = [c for c in all_cases if c not in set(val_cases)]
        splits.append({"train": sorted(train_cases), "val": sorted(val_cases)})

    return splits


def generate(config_path: str):
    cfg = load_config(config_path)

    dataset_id   = int(cfg["dataset_id"])
    dataset_name = cfg["dataset_name"]
    preprocessed = Path(cfg["nnunet_preprocessed"])
    data_dir     = cfg["data_dir"]
    json_list    = cfg["json_list"]

    splits = read_folds(json_list, data_dir)

    out_dir = preprocessed / f"Dataset{dataset_id:03d}_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "splits_final.json"

    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Written {len(splits)} folds to {out_path}")
    for i, split in enumerate(splits):
        print(f"  Fold {i}: {len(split['train'])} train, {len(split['val'])} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    args = parser.parse_args()
    generate(args.config)

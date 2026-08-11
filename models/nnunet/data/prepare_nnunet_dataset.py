"""
Convert BraTS-GoAT data to the nnUNet raw-data folder structure.

nnUNet expects:
  nnUNet_raw/
    Dataset{id:03d}_{name}/
      imagesTr/   {case}_0000.nii.gz  (T1c)
                  {case}_0001.nii.gz  (T1n)
                  {case}_0002.nii.gz  (T2w)
                  {case}_0003.nii.gz  (T2f)
      labelsTr/   {case}.nii.gz       (integer seg: 0=bg, 1=NCR, 2=SNFH, 3=ET)
      dataset.json

Label convention (BraTS-GoAT):
  0 = Background
  1 = NCR  (Non-Contrast-enhancing Tumour core)
  2 = SNFH (Surrounding Non-enhancing FLAIR Hyperintensity)
  3 = ET   (Enhancing Tumour)

Region metrics are derived in evaluate.py:
  ET  = voxels with label 3
  TC  = voxels with labels {1, 3}
  WT  = voxels with labels {1, 2, 3}

Usage
-----
python data/prepare_nnunet_dataset.py \
    --config configs/paths.yaml \
    [--copy]          # copy files instead of symlinking (slower but portable)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def read_brats_json(json_list: str, data_dir: str) -> list[dict]:
    """Return list of {case_id, t1c, t1n, t2w, t2f, seg, fold}."""
    with open(json_list) as f:
        raw = json.load(f)

    entries = []
    for item in raw["training"]:
        images = item["image"]   # [t1c, t1n, t2w, t2f] relative paths
        label  = item["label"]
        fold   = item.get("fold", -1)

        # Resolve to absolute paths
        def abs_path(p):
            return str(Path(data_dir) / p) if not Path(p).is_absolute() else p

        t1c, t1n, t2w, t2f = [abs_path(p) for p in images]
        seg = abs_path(label)

        # Derive a clean case identifier from the subject folder name
        case_id = Path(t1c).parent.name  # e.g. "BraTS-GLI-00000-000"

        entries.append(dict(case_id=case_id, t1c=t1c, t1n=t1n, t2w=t2w, t2f=t2f, seg=seg, fold=fold))

    return entries


def link_or_copy(src: str, dst: str, use_copy: bool):
    dst_path = Path(dst)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    if use_copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def build_dataset_json(entries: list[dict], dataset_name: str, out_path: str):
    dataset = {
        "channel_names": {
            "0": "T1c",
            "1": "T1n",
            "2": "T2w",
            "3": "T2f",
        },
        "labels": {
            "background": 0,
            "NCR": 1,
            "SNFH": 2,
            "ET": 3,
        },
        "numTraining": len(entries),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "BraTS-GoAT Task 3",
        "reference": "https://www.synapse.org/brats2026",
        "licence": "see BraTS challenge",
        "release": "2026",
    }
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"  Written dataset.json → {out_path}")


def prepare(config_path: str, use_copy: bool = False):
    cfg = load_config(config_path)

    dataset_id   = int(cfg["dataset_id"])
    dataset_name = cfg["dataset_name"]
    nnunet_raw   = Path(cfg["nnunet_raw"])
    data_dir     = cfg["data_dir"]
    json_list    = cfg["json_list"]
    test_fold    = int(cfg.get("test_fold", 5))

    dataset_folder = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    images_ts = dataset_folder / "imagesTs"

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    entries = read_brats_json(json_list, data_dir)
    print(f"Found {len(entries)} cases in {json_list}")

    action = "Copying" if use_copy else "Symlinking"
    train_entries = [e for e in entries if e["fold"] != test_fold]
    test_entries  = [e for e in entries if e["fold"] == test_fold]

    for entry in train_entries:
        cid = entry["case_id"]
        for suffix, src_key in [("_0000", "t1c"), ("_0001", "t1n"), ("_0002", "t2w"), ("_0003", "t2f")]:
            dst = images_tr / f"{cid}{suffix}.nii.gz"
            link_or_copy(entry[src_key], str(dst), use_copy)
        dst_seg = labels_tr / f"{cid}.nii.gz"
        link_or_copy(entry["seg"], str(dst_seg), use_copy)

    print(f"{action} done for {len(train_entries)} training cases.")

    for entry in test_entries:
        cid = entry["case_id"]
        for suffix, src_key in [("_0000", "t1c"), ("_0001", "t1n"), ("_0002", "t2w"), ("_0003", "t2f")]:
            dst = images_ts / f"{cid}{suffix}.nii.gz"
            link_or_copy(entry[src_key], str(dst), use_copy)
        # GT labels stay in labelsTr so evaluate.py can find them
        dst_seg = labels_tr / f"{cid}.nii.gz"
        link_or_copy(entry["seg"], str(dst_seg), use_copy)

    print(f"{action} done for {len(test_entries)} test cases (fold {test_fold}).")
    build_dataset_json(train_entries, dataset_name, str(dataset_folder / "dataset.json"))
    print(f"\nDataset ready at: {dataset_folder}")
    print(f"Next step: python src/main.py --step preprocess  (or run scripts/01_plan_and_preprocess.sh)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    args = parser.parse_args()
    prepare(args.config, use_copy=args.copy)

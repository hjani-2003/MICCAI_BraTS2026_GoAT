"""
nnUNet BraTS pipeline orchestrator.

Wraps the four nnUNet v2 CLI commands so the full pipeline can be driven
from one place.  Each step can also be run standalone via the shell scripts
in scripts/.

Steps
-----
  prepare     -> data/prepare_nnunet_dataset.py   (BraTS -> nnUNet raw format)
  splits      -> data/generate_nnunet_splits.py    (sync fold splits)
  preprocess  -> nnUNetv2_plan_and_preprocess
  train       -> nnUNetv2_train  (one fold, or all folds with --all-folds)
  predict     -> nnUNetv2_predict
  evaluate    -> scripts/evaluate.py

Usage
-----
# Full pipeline, fold 0:
python src/main.py --steps prepare splits preprocess train predict evaluate --fold 0

# Just train fold 2 (assuming preprocess already done):
python src/main.py --steps train --fold 2

# Evaluate predictions already on disk:
python src/main.py --steps evaluate --fold 0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str) -> dict:
    with open(REPO_ROOT / config_path) as f:
        return yaml.safe_load(f)


def set_nnunet_env(cfg: dict):
    """Export the three required nnUNet environment variables."""
    os.environ["nnUNet_raw"]          = cfg["nnunet_raw"]
    os.environ["nnUNet_preprocessed"] = cfg["nnunet_preprocessed"]
    os.environ["nnUNet_results"]      = cfg["nnunet_results"]



def run(cmd: list[str], **kwargs):
    print(f"\n$ {' '.join(cmd)}\n{'─'*60}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def step_prepare(cfg: dict, use_copy: bool):
    run([
        sys.executable, str(REPO_ROOT / "data" / "prepare_nnunet_dataset.py"),
        "--config", str(REPO_ROOT / "configs" / "paths.yaml"),
        *(["--copy"] if use_copy else []),
    ])


def step_splits(cfg: dict):
    run([
        sys.executable, str(REPO_ROOT / "data" / "generate_nnunet_splits.py"),
        "--config", str(REPO_ROOT / "configs" / "paths.yaml"),
    ])


def step_preprocess(cfg: dict):
    dataset_id = int(cfg["dataset_id"])
    run([
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity",
    ])


def step_train(cfg: dict, fold: int):
    dataset_id    = int(cfg["dataset_id"])
    configuration = cfg["configuration"]
    trainer       = cfg.get("trainer", "nnUNetTrainer")
    run([
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-tr", trainer,
    ])


def step_validate(cfg: dict, fold: int):
    dataset_id    = int(cfg["dataset_id"])
    configuration = cfg["configuration"]
    trainer       = cfg.get("trainer", "nnUNetTrainer")
    run([
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-tr", trainer,
        "--val",
        "--npz",
    ])


def step_predict(cfg: dict, fold: int, pred_output: str | None):
    dataset_id    = int(cfg["dataset_id"])
    dataset_name  = cfg["dataset_name"]
    configuration = cfg["configuration"]
    trainer       = cfg.get("trainer", "nnUNetTrainer")
    nnunet_raw    = cfg["nnunet_raw"]
    nnunet_results = cfg["nnunet_results"]

    images_tr = str(Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{dataset_name}" / "imagesTr")

    if pred_output is None:
        pred_output = str(
            Path(nnunet_results)
            / f"Dataset{dataset_id:03d}_{dataset_name}"
            / f"{trainer}__nnUNetPlans__{configuration}"
            / f"fold_{fold}"
            / "validation"
        )

    run([
        "nnUNetv2_predict",
        "-i",      images_tr,
        "-o",      pred_output,
        "-d",      str(dataset_id),
        "-c",      configuration,
        "-tr",     trainer,
        "-f",      str(fold),
        "--save_probabilities",  # keep for ensemble if needed
    ])
    return pred_output


def step_predict_test(cfg: dict, fold: int, pred_output: str | None):
    dataset_id    = int(cfg["dataset_id"])
    dataset_name  = cfg["dataset_name"]
    configuration = cfg["configuration"]
    trainer       = cfg.get("trainer", "nnUNetTrainer")
    nnunet_raw    = cfg["nnunet_raw"]
    nnunet_results = cfg["nnunet_results"]

    images_ts = str(Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{dataset_name}" / "imagesTs")

    if pred_output is None:
        pred_output = str(
            Path(nnunet_results)
            / f"Dataset{dataset_id:03d}_{dataset_name}"
            / f"{trainer}__nnUNetPlans__{configuration}"
            / f"fold_{fold}"
            / "test_predictions"
        )

    run([
        "nnUNetv2_predict",
        "-i",  images_ts,
        "-o",  pred_output,
        "-d",  str(dataset_id),
        "-c",  configuration,
        "-tr", trainer,
        "-f",  str(fold),
        "--save_probabilities",
    ])
    return pred_output


def step_evaluate_test(cfg: dict, fold: int, pred_dir: str | None, output_csv: str | None):
    dataset_id    = int(cfg["dataset_id"])
    dataset_name  = cfg["dataset_name"]
    nnunet_raw    = cfg["nnunet_raw"]
    trainer       = cfg.get("trainer", "nnUNetTrainer")
    configuration = cfg["configuration"]
    nnunet_results = cfg["nnunet_results"]
    test_fold     = int(cfg.get("test_fold", 5))

    label_dir = str(Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{dataset_name}" / "labelsTr")

    if pred_dir is None:
        pred_dir = str(
            Path(nnunet_results)
            / f"Dataset{dataset_id:03d}_{dataset_name}"
            / f"{trainer}__nnUNetPlans__{configuration}"
            / f"fold_{fold}"
            / "test_predictions"
        )

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--pred-dir",  pred_dir,
        "--label-dir", label_dir,
        "--fold-json", cfg["json_list"],
        "--fold",      str(test_fold),
    ]
    if output_csv:
        cmd += ["--output", output_csv]

    run(cmd)


def step_prepare_goat(cfg: dict, use_copy: bool = False):
    """
    Symlink (or copy) GoAT validation modalities into nnUNet input format.

    Source layout : {goat_val_dir}/BraTS-GoAT-XXXXX/{case}-t1c.nii.gz …
    Output layout : {nnunet_raw}/Dataset001_BraTS/imagesGoAT/{case}_0000.nii.gz …

    Channel order matches the trained model (same as imagesTr):
      _0000 = T1c,  _0001 = T1n,  _0002 = T2w,  _0003 = T2f
    """
    import shutil

    goat_val_dir = cfg.get("goat_val_dir")
    if not goat_val_dir:
        print("ERROR: 'goat_val_dir' not set in configs/paths.yaml")
        sys.exit(1)

    dataset_id   = int(cfg["dataset_id"])
    dataset_name = cfg["dataset_name"]
    nnunet_raw   = Path(cfg["nnunet_raw"])
    images_goat  = nnunet_raw / f"Dataset{dataset_id:03d}_{dataset_name}" / "imagesGoAT"
    images_goat.mkdir(parents=True, exist_ok=True)

    goat_root = Path(goat_val_dir)
    modality_map = [("t1c", "_0000"), ("t1n", "_0001"), ("t2w", "_0002"), ("t2f", "_0003")]

    cases = sorted(p for p in goat_root.iterdir() if p.is_dir())
    print(f"Found {len(cases)} GoAT cases in {goat_root}")

    for folder in cases:
        cid = folder.name  # e.g. BraTS-GoAT-02489
        for suffix, channel in modality_map:
            src = folder / f"{cid}-{suffix}.nii.gz"
            dst = images_goat / f"{cid}{channel}.nii.gz"
            if not src.exists():
                print(f"WARNING: missing {src} — skipping case {cid}")
                break
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if use_copy:
                shutil.copy2(src, dst)
            else:
                os.symlink(src.resolve(), dst)

    action = "Copied" if use_copy else "Symlinked"
    print(f"{action} {len(cases)} GoAT cases → {images_goat}")


def step_predict_goat(cfg: dict, fold: int, pred_output: str | None, save_probabilities: bool = True):
    """Run nnUNetv2_predict on imagesGoAT (hard labels always; probabilities optional)."""
    dataset_id    = int(cfg["dataset_id"])
    dataset_name  = cfg["dataset_name"]
    configuration = cfg["configuration"]
    trainer       = cfg.get("trainer", "nnUNetTrainer")
    nnunet_raw    = cfg["nnunet_raw"]
    nnunet_results = cfg["nnunet_results"]

    images_goat = str(Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{dataset_name}" / "imagesGoAT")

    if pred_output is None:
        pred_output = str(
            Path(nnunet_results)
            / f"Dataset{dataset_id:03d}_{dataset_name}"
            / f"{trainer}__nnUNetPlans__{configuration}"
            / f"fold_{fold}"
            / "goat_val_predictions"
        )

    cmd = [
        "nnUNetv2_predict",
        "-i",  images_goat,
        "-o",  pred_output,
        "-d",  str(dataset_id),
        "-c",  configuration,
        "-tr", trainer,
        "-f",  str(fold),
    ]
    if save_probabilities:
        cmd.append("--save_probabilities")
    run(cmd)
    return pred_output


def step_evaluate(cfg: dict, fold: int, pred_dir: str | None, output_csv: str | None):
    dataset_id   = int(cfg["dataset_id"])
    dataset_name = cfg["dataset_name"]
    nnunet_raw   = cfg["nnunet_raw"]
    trainer      = cfg.get("trainer", "nnUNetTrainer")
    configuration = cfg["configuration"]
    nnunet_results = cfg["nnunet_results"]

    label_dir = str(Path(nnunet_raw) / f"Dataset{dataset_id:03d}_{dataset_name}" / "labelsTr")

    if pred_dir is None:
        pred_dir = str(
            Path(nnunet_results)
            / f"Dataset{dataset_id:03d}_{dataset_name}"
            / f"{trainer}__nnUNetPlans__{configuration}"
            / f"fold_{fold}"
            / "validation"
        )

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--pred-dir",  pred_dir,
        "--label-dir", label_dir,
        "--fold-json", cfg["json_list"],
        "--fold",      str(fold),
    ]
    if output_csv:
        cmd += ["--output", output_csv]

    run(cmd)


def main():
    parser = argparse.ArgumentParser(description="nnUNet BraTS pipeline")
    parser.add_argument("--config",  default="configs/paths.yaml")
    parser.add_argument("--steps",   nargs="+",
                        choices=["prepare", "splits", "preprocess", "train", "validate", "predict", "evaluate",
                                 "predict_test", "evaluate_test",
                                 "prepare_goat", "predict_goat"],
                        default=["train"])
    parser.add_argument("--fold",    type=int, default=0)
    parser.add_argument("--all-folds", action="store_true", help="Train / predict all configured folds.")
    parser.add_argument("--copy",    action="store_true", help="Copy data instead of symlinking.")
    parser.add_argument("--pred-dir", default=None, help="Override prediction output directory.")
    parser.add_argument("--output-csv", default=None, help="Path for per-case CSV from evaluate step.")
    parser.add_argument("--hard-labels-only", action="store_true",
                         help="Skip --save_probabilities for predict_goat (hard-label .nii.gz only).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_nnunet_env(cfg)

    folds = cfg.get("folds", [0, 1, 2, 3, 4]) if args.all_folds else [args.fold]

    for step in args.steps:
        if step == "prepare":
            step_prepare(cfg, args.copy)
        elif step == "splits":
            step_splits(cfg)
        elif step == "preprocess":
            step_preprocess(cfg)
        elif step == "train":
            for fold in folds:
                step_train(cfg, fold)
        elif step == "validate":
            for fold in folds:
                step_validate(cfg, fold)
        elif step == "predict":
            for fold in folds:
                step_predict(cfg, fold, args.pred_dir)
        elif step == "evaluate":
            for fold in folds:
                step_evaluate(cfg, fold, args.pred_dir, args.output_csv)
        elif step == "predict_test":
            step_predict_test(cfg, args.fold, args.pred_dir)
        elif step == "evaluate_test":
            step_evaluate_test(cfg, args.fold, args.pred_dir, args.output_csv)
        elif step == "prepare_goat":
            step_prepare_goat(cfg, args.copy)
        elif step == "predict_goat":
            for fold in folds:
                step_predict_goat(cfg, fold, args.pred_dir, save_probabilities=not args.hard_labels_only)


if __name__ == "__main__":
    main()

# STANDARD LIBRARY
import argparse
import datetime
import json
import os
import sys
import zipfile
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# MONAI IMPORTS
from monai import data, transforms
from monai.inferers import sliding_window_inference

# CUSTOM IMPORTS
from src.main import REPO_ROOT, build_model, load_yaml, resolve_path


DEFAULT_VAL_DATA_DIR = "/path/to/BraTS-GoAT-ValidationData"

# Channel order the models were trained on (data/json_list_conversion.py: image = [t1c, t1n, t2f, t2w]).
MODALITY_SUFFIXES = ["t1c", "t1n", "t2f", "t2w"]

# scheme="regions" model output channel order (scripts/utils/custom_transforms.py:ConvertToBraTSRegionsd).
REGION_NAMES = ["ET", "TC", "WT"]

# mini_detector_gpu sibling repo, used as an opt-in post-processing fallback
# for cases where WT/TC/ET comes out completely empty (see
# postprocess_empty_regions below).
MINI_DETECTOR_ROOT = REPO_ROOT.parent / "mini_detector_gpu"
DEFAULT_MINI_DETECTOR_MODEL_CONFIG = str(MINI_DETECTOR_ROOT / "configs" / "model.yaml")


def build_predict_transform(k_div):
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.CropForegroundd(
                keys=["image"],
                source_key="image",
                k_divisible=k_div,
                allow_smaller=True,
            ),
        ]
    )


def gather_validation_cases(data_dir):
    """Walk a BraTS-GoAT-style validation directory: one folder per case, each
    containing <case>-t1c/t1n/t2f/t2w.nii.gz and no ground-truth segmentation."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data_dir does not exist: {data_dir}")

    cases = []
    for case_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        case_id = case_dir.name
        image_paths = [str(case_dir / f"{case_id}-{suffix}.nii.gz") for suffix in MODALITY_SUFFIXES]
        missing = [p for p in image_paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Case {case_id!r} is missing modality file(s): {missing}")
        cases.append({"image": image_paths, "case_id": case_id})
    return cases


def filter_cases(cases, case_id):
    """Restrict `cases` (as returned by gather_validation_cases) to a single case id."""
    if case_id is None:
        return cases
    filtered = [c for c in cases if c["case_id"] == case_id]
    if not filtered:
        available = ", ".join(c["case_id"] for c in cases[:10])
        suffix = ", ..." if len(cases) > 10 else ""
        raise ValueError(f"Case {case_id!r} not found. First case ids available: {available}{suffix}")
    return filtered


def scheme_from_out_channels(out_channels: int) -> str:
    if out_channels == 4:
        return "multiclass"
    if out_channels == 3:
        return "regions"
    raise ValueError(
        f"Cannot infer prediction scheme from model out_channels={out_channels}; pass --scheme explicitly."
    )


def out_channels_for_scheme(scheme: str) -> int:
    if scheme == "multiclass":
        return 4
    if scheme == "regions":
        return 3
    raise ValueError(f"Unknown scheme: {scheme!r}")


def resolve_scheme_and_model_config(model_config: dict, scheme: str | None):


    model_cfg = model_config.get("model", {})
    config_out_channels = int(model_cfg.get("out_channels", 4))
    resolved_scheme = scheme or scheme_from_out_channels(config_out_channels)
    if resolved_scheme not in {"multiclass", "regions"}:
        raise ValueError(f"--scheme must be 'multiclass' or 'regions', got {resolved_scheme!r}")

    required_out_channels = out_channels_for_scheme(resolved_scheme)
    if required_out_channels != config_out_channels:
        print(
            f"Note: --scheme={resolved_scheme!r} requires out_channels={required_out_channels}, "
            f"but model-config has out_channels={config_out_channels}. Building the model with "
            f"out_channels={required_out_channels} to match the checkpoint's actual architecture."
        )
        model_config = {**model_config, "model": {**model_cfg, "out_channels": required_out_channels}}

    return resolved_scheme, required_out_channels, model_config


def clamp_region_masks(et_mask, tc_mask, wt_mask):


    tc_mask = tc_mask | et_mask
    wt_mask = wt_mask | tc_mask
    return et_mask, tc_mask, wt_mask


def resolve_region_thresholds(threshold) -> dict:


    if isinstance(threshold, dict):
        missing = [r for r in REGION_NAMES if r not in threshold]
        if missing:
            raise ValueError(f"threshold dict is missing region(s): {missing}")
        return {r: float(threshold[r]) for r in REGION_NAMES}
    return {r: float(threshold) for r in REGION_NAMES}


def regions_to_multiclass(et_prob, tc_prob, wt_prob, threshold=0.5):


    thr = resolve_region_thresholds(threshold)

    wt_mask = wt_prob > thr["WT"]
    tc_mask = tc_prob > thr["TC"]
    et_mask = et_prob > thr["ET"]

    et_mask, tc_mask, wt_mask = clamp_region_masks(et_mask, tc_mask, wt_mask)

    seg = torch.zeros_like(wt_prob, dtype=torch.long)
    seg[wt_mask] = 2  # ED (WT shell)
    seg[tc_mask] = 1  # NCR (TC minus ET, refined next)
    seg[et_mask] = 3  # ET
    return seg


def probs_to_label_map(probs: torch.Tensor, scheme: str, threshold=0.5) -> torch.Tensor:


    if scheme == "multiclass":
        return torch.argmax(probs, dim=0)
    if scheme == "regions":
        return regions_to_multiclass(probs[0], probs[1], probs[2], threshold=threshold)
    raise ValueError(f"Unknown scheme: {scheme!r}")


def load_model(checkpoint_path, model_config, device):
    model = build_model(model_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def pred_arr_to_regions(pred_arr):
    """Inverse of regions_to_multiclass: derive independent WT/TC/ET boolean
    masks from a mutually-exclusive label map (0=background, 1=NCR, 2=ED, 3=ET)."""
    et_mask = pred_arr == 3
    tc_mask = (pred_arr == 1) | (pred_arr == 3)
    wt_mask = pred_arr > 0
    return et_mask, tc_mask, wt_mask


def load_mini_detector(checkpoint_path, model_config_path, device):
    
    if not MINI_DETECTOR_ROOT.is_dir():
        raise NotADirectoryError(f"mini_detector_gpu repo not found at {MINI_DETECTOR_ROOT}")
    if str(MINI_DETECTOR_ROOT) not in sys.path:
        sys.path.insert(0, str(MINI_DETECTOR_ROOT))
    import infer as mini_detector_infer

    model_config = load_yaml(resolve_path(model_config_path))
    model = mini_detector_infer.load_model(checkpoint_path, model_config, device)
    return mini_detector_infer, model


def postprocess_empty_regions(pred_arr, entry, mini_detector_infer, mini_detector_model, device, threshold=0.5):
    
    et_mask, tc_mask, wt_mask = pred_arr_to_regions(pred_arr)
    empty_regions = [
        name for name, mask in (("ET", et_mask), ("TC", tc_mask), ("WT", wt_mask)) if mask.sum() == 0
    ]
    if not empty_regions:
        return pred_arr

    case_dir = Path(entry["image"][0]).parent
    mini_image_paths = mini_detector_infer.image_paths_for_case(case_dir, entry["case_id"])
    md_probs = mini_detector_infer.predict_regions(mini_detector_model, mini_image_paths, device)

    if md_probs.shape[1:] != pred_arr.shape:
        raise ValueError(
            f"mini_detector_gpu output shape {md_probs.shape[1:]} does not match "
            f"prediction shape {pred_arr.shape} for case {entry['case_id']!r}."
        )

    md_et_mask, md_tc_mask, md_wt_mask = md_probs[0] > threshold, md_probs[1] > threshold, md_probs[2] > threshold

    final_et = md_et_mask if "ET" in empty_regions else et_mask
    final_tc = md_tc_mask if "TC" in empty_regions else tc_mask
    final_wt = md_wt_mask if "WT" in empty_regions else wt_mask

    print(f"  mini_detector_gpu fallback for {entry['case_id']}: replacing empty region(s) {empty_regions}")

    seg = regions_to_multiclass(
        torch.as_tensor(final_et.astype(np.float32)),
        torch.as_tensor(final_tc.astype(np.float32)),
        torch.as_tensor(final_wt.astype(np.float32)),
        threshold=0.5,
    )
    return seg.numpy()


def save_prediction(pred_arr, affine, header, case_id, output_dir, save_npz):
    os.makedirs(output_dir, exist_ok=True)
    pred_arr = pred_arr.astype(np.uint8)

    nii_path = os.path.join(output_dir, f"{case_id}.nii.gz")
    nib.save(nib.Nifti1Image(pred_arr, affine=affine, header=header), nii_path)

    if save_npz:
        npz_path = os.path.join(output_dir, f"{case_id}.npz")
        np.savez_compressed(npz_path, pred=pred_arr, affine=affine)


def zip_submission(output_dir, zip_path=None):
    """Zip all .nii.gz predictions in output_dir (flat, no subfolders) for submission."""
    nii_files = sorted(Path(output_dir).glob("*.nii.gz"))
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz predictions found in {output_dir} to zip.")

    zip_path = resolve_path(zip_path) if zip_path else os.path.join(output_dir, "submission.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for nii_file in nii_files:
            zf.write(nii_file, arcname=nii_file.name)

    print(f"Submission zip written to: {zip_path}")
    return zip_path


def run_inference(
    train_config_path: str,
    model_config_path: str,
    checkpoint_path: str,
    data_dir: str | None = None,
    case_id: str | None = None,
    output_dir: str | None = None,
    scheme: str | None = None,
    threshold: float | dict = 0.5,
    infer_overlap: float | None = None,
    sw_batch_size: int | None = None,
    num_workers: int | None = None,
    save_npz: bool = False,
    make_zip: bool = False,
    zip_path: str | None = None,
    mini_detector_checkpoint: str | None = None,
    mini_detector_model_config: str | None = None,
    mini_detector_threshold: float = 0.5,
):
    
    train_config = load_yaml(resolve_path(train_config_path))
    model_config = load_yaml(resolve_path(model_config_path))

    data_cfg = train_config.get("data", {})

    roi = list(data_cfg.get("roi", [128, 128, 128]))
    min_dims = [32, 32, 32]

    scheme, out_channels, model_config = resolve_scheme_and_model_config(model_config, scheme)

    data_dir = resolve_path(data_dir) if data_dir else resolve_path(DEFAULT_VAL_DATA_DIR)
    infer_overlap = float(data_cfg.get("infer_overlap", 0.5) if infer_overlap is None else infer_overlap)
    sw_batch_size = int(data_cfg.get("sw_batch_size", 1) if sw_batch_size is None else sw_batch_size)
    num_workers = int(data_cfg.get("num_workers", 4) if num_workers is None else num_workers)

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        output_dir = os.path.join(REPO_ROOT, "outputs", "predictions", timestamp)
    else:
        output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = resolve_path(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Predicting with checkpoint {checkpoint_path!r}, scheme={scheme!r} (out_channels={out_channels})")
    model = load_model(checkpoint_path, model_config, device)

    mini_detector_infer_mod, mini_detector_model = None, None
    if mini_detector_checkpoint is not None:
        mini_detector_checkpoint = resolve_path(mini_detector_checkpoint)
        mini_detector_model_config = mini_detector_model_config or DEFAULT_MINI_DETECTOR_MODEL_CONFIG
        print(f"Loading mini_detector_gpu fallback checkpoint {mini_detector_checkpoint!r}")
        mini_detector_infer_mod, mini_detector_model = load_mini_detector(
            mini_detector_checkpoint, mini_detector_model_config, device
        )

    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=infer_overlap,
    )

    cases = filter_cases(gather_validation_cases(data_dir), case_id)
    if len(cases) == 0:
        raise RuntimeError(f"No cases found in {data_dir}")

    pre_transform = build_predict_transform(k_div=min_dims)
    post_transform = transforms.Invertd(
        keys="pred",
        transform=pre_transform,
        orig_keys="image",
        nearest_interp=False,
        to_tensor=True,
    )

    predict_ds = data.Dataset(data=cases, transform=pre_transform)
    predict_loader = data.DataLoader(
        predict_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    with torch.no_grad():
        for entry, batch_data in zip(cases, predict_loader):
            case_id_ = entry["case_id"]
            image = batch_data["image"].to(device)

            logits = model_inferer(image).float()
            probs = torch.softmax(logits, dim=1) if scheme == "multiclass" else torch.sigmoid(logits)
            probs = probs.cpu()

            batch_data["pred"] = probs
            item = data.decollate_batch(batch_data)[0]
            item = post_transform(item)

            probs_arr = item["pred"]
            if hasattr(probs_arr, "as_tensor"):
                probs_arr = probs_arr.as_tensor()
            if not torch.is_tensor(probs_arr):
                probs_arr = torch.as_tensor(np.asarray(probs_arr))

            label_map = probs_to_label_map(probs_arr, scheme=scheme, threshold=threshold)
            pred_arr = label_map.cpu().numpy()

            if mini_detector_model is not None:
                pred_arr = postprocess_empty_regions(
                    pred_arr, entry, mini_detector_infer_mod, mini_detector_model, device,
                    threshold=mini_detector_threshold,
                )

            original_image_path = entry["image"][0]
            original_nii = nib.load(original_image_path)

            save_prediction(
                pred_arr=pred_arr,
                affine=original_nii.affine,
                header=original_nii.header,
                case_id=case_id_,
                output_dir=output_dir,
                save_npz=save_npz,
            )
            print(f"Saved prediction for {case_id_} -> {output_dir}")

    print(f"Done. Predictions written to: {output_dir}")

    if make_zip:
        zip_submission(output_dir, zip_path=zip_path)

    return output_dir


def add_region_threshold_cli_args(parser):
    
    parser.add_argument("--threshold", type=float, default=0.5, help="Blanket fallback sigmoid threshold used only for --scheme regions, for any region not covered by --best-thresholds or its own --threshold-* override.")
    parser.add_argument("--best-thresholds", default=None, help="Path to a best_thresholds.json written by src/tune_thresholds.py; its per-region thresholds become the default for --threshold-et/-tc/-wt (still overridable individually).")
    parser.add_argument("--threshold-et", type=float, default=None, help="Per-region override of --threshold/--best-thresholds for ET.")
    parser.add_argument("--threshold-tc", type=float, default=None, help="Per-region override of --threshold/--best-thresholds for TC.")
    parser.add_argument("--threshold-wt", type=float, default=None, help="Per-region override of --threshold/--best-thresholds for WT.")


def load_best_thresholds(path) -> dict:
    
    with open(resolve_path(path)) as f:
        data = json.load(f)
    tuned = data.get("threshold", data)
    missing = [r for r in REGION_NAMES if r not in tuned]
    if missing:
        raise ValueError(f"{path!r} is missing region threshold(s): {missing}")
    return {r: float(tuned[r]) for r in REGION_NAMES}


def region_threshold_from_args(args) -> dict:
    
    thresholds = {r: args.threshold for r in REGION_NAMES}

    best_thresholds_path = getattr(args, "best_thresholds", None)
    if best_thresholds_path:
        thresholds.update(load_best_thresholds(best_thresholds_path))

    overrides = {"ET": args.threshold_et, "TC": args.threshold_tc, "WT": args.threshold_wt}
    for region, value in overrides.items():
        if value is not None:
            thresholds[region] = value

    return thresholds


def cli():
    parser = argparse.ArgumentParser(description="Run single-checkpoint inference for a trained SwinUNETR model.")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Path to the training config YAML.")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to the model config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--data-dir", default=None, help=f"Validation data directory. Default: {DEFAULT_VAL_DATA_DIR}")
    parser.add_argument("--case-id", default=None, help="Predict a single case by id (e.g. BraTS-GoAT-02489). Default: every case in data-dir.")
    parser.add_argument("--output-dir", default=None, help="Directory to write predictions. Auto-generated under outputs/predictions/<timestamp> if omitted.")
    parser.add_argument(
        "--scheme",
        choices=["multiclass", "regions"],
        default=None,
        help=(
            "How to interpret model output channels. 'multiclass' = 4-channel softmax "
            "[bg, NCR, ED, ET] (argmax). 'regions' = 3-channel sigmoid [ET, TC, WT] "
            "(nested-threshold). Inferred from model-config out_channels if omitted."
        ),
    )
    add_region_threshold_cli_args(parser)
    parser.add_argument("--infer-overlap", type=float, default=None, help="Sliding window inference overlap override.")
    parser.add_argument("--sw-batch-size", type=int, default=None, help="Sliding window batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader worker count override.")
    parser.add_argument("--npz", action="store_true", help="Additionally save each prediction as a .npz file.")
    parser.add_argument("--zip", dest="make_zip", action="store_true", help="Zip the .nii.gz predictions into a submission archive.")
    parser.add_argument("--zip-path", default=None, help="Custom path for the submission zip (default: <output_dir>/submission.zip).")
    parser.add_argument(
        "--mini-detector-checkpoint",
        default=None,
        help=(
            "Path to a mini_detector_gpu (SmallTumorHRSeg) checkpoint. If set, enables a "
            "post-processing fallback: for any case where WT/TC/ET comes out completely "
            "empty, that region is replaced with mini_detector_gpu's own prediction. "
            "Disabled (no effect on output) if omitted."
        ),
    )
    parser.add_argument(
        "--mini-detector-model-config",
        default=None,
        help=f"Path to mini_detector_gpu's model config YAML. Default: {DEFAULT_MINI_DETECTOR_MODEL_CONFIG}",
    )
    parser.add_argument(
        "--mini-detector-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold used to binarize mini_detector_gpu's region predictions.",
    )

    args = parser.parse_args()

    run_inference(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        case_id=args.case_id,
        output_dir=args.output_dir,
        scheme=args.scheme,
        threshold=region_threshold_from_args(args),
        infer_overlap=args.infer_overlap,
        sw_batch_size=args.sw_batch_size,
        num_workers=args.num_workers,
        save_npz=args.npz,
        make_zip=args.make_zip,
        zip_path=args.zip_path,
        mini_detector_checkpoint=args.mini_detector_checkpoint,
        mini_detector_model_config=args.mini_detector_model_config,
        mini_detector_threshold=args.mini_detector_threshold,
    )


if __name__ == "__main__":
    cli()

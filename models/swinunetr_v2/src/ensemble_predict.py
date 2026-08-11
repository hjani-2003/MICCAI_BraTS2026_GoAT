# STANDARD LIBRARY
import argparse
import datetime
import os
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# MONAI IMPORTS
from monai import data, transforms
from monai.inferers import sliding_window_inference

# CUSTOM IMPORTS
from src.main import REPO_ROOT, load_yaml, resolve_path
from src.predict import (
    DEFAULT_MINI_DETECTOR_MODEL_CONFIG,
    DEFAULT_VAL_DATA_DIR,
    add_region_threshold_cli_args,
    build_predict_transform,
    filter_cases,
    gather_validation_cases,
    load_mini_detector,
    load_model,
    postprocess_empty_regions,
    probs_to_label_map,
    region_threshold_from_args,
    resolve_scheme_and_model_config,
    save_prediction,
    zip_submission,
)


def resolve_checkpoints(checkpoint_dir, checkpoint_paths):
    if checkpoint_paths:
        return [resolve_path(p) for p in checkpoint_paths]
    checkpoint_dir = resolve_path(checkpoint_dir)
    found = sorted(set(Path(checkpoint_dir).glob("*.pth")) | set(Path(checkpoint_dir).glob("*.pt")))
    if not found:
        raise FileNotFoundError(f"No .pth/.pt checkpoints found in {checkpoint_dir}")
    return [str(p) for p in found]


def run_ensemble_inference(
    train_config_path: str,
    model_config_path: str,
    checkpoint_dir: str | None = None,
    checkpoints: list[str] | None = None,
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
    """N-fold ensemble inference: averages per-voxel probabilities across every
    checkpoint before converting to the submission label map. For a single
    checkpoint, use src/predict.py instead."""
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths = resolve_checkpoints(checkpoint_dir or "./checkpoints", checkpoints)
    print(f"Ensembling {len(checkpoint_paths)} checkpoint(s), scheme={scheme!r} (out_channels={out_channels}):")
    for p in checkpoint_paths:
        print(f"  - {p}")

    models = [load_model(p, model_config, device) for p in checkpoint_paths]

    mini_detector_infer_mod, mini_detector_model = None, None
    if mini_detector_checkpoint is not None:
        mini_detector_checkpoint = resolve_path(mini_detector_checkpoint)
        mini_detector_model_config = mini_detector_model_config or DEFAULT_MINI_DETECTOR_MODEL_CONFIG
        print(f"Loading mini_detector_gpu fallback checkpoint {mini_detector_checkpoint!r}")
        mini_detector_infer_mod, mini_detector_model = load_mini_detector(
            mini_detector_checkpoint, mini_detector_model_config, device
        )

    model_inferers = [
        partial(
            sliding_window_inference,
            roi_size=roi,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=infer_overlap,
        )
        for model in models
    ]

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

            avg_probs = None
            for inferer in model_inferers:
                logits = inferer(image).float()
                probs = torch.softmax(logits, dim=1) if scheme == "multiclass" else torch.sigmoid(logits)
                avg_probs = probs if avg_probs is None else avg_probs + probs
            avg_probs = (avg_probs / len(model_inferers)).cpu()

            batch_data["pred"] = avg_probs
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
            print(f"Saved ensemble prediction for {case_id_} -> {output_dir}")

    print(f"Done. Predictions written to: {output_dir}")

    if make_zip:
        zip_submission(output_dir, zip_path=zip_path)

    return output_dir


def cli():
    parser = argparse.ArgumentParser(description="Run N-fold ensemble inference for SwinUNETR checkpoints.")
    parser.add_argument("--train-config", default="configs/train.yaml", help="Path to the training config YAML.")
    parser.add_argument("--model-config", default="configs/model.yaml", help="Path to the model config YAML.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory containing fold checkpoints (globs *.pth/*.pt).")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="Explicit checkpoint paths; overrides --checkpoint-dir.")
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
            "(nested-threshold). Inferred from model-config out_channels if omitted. "
            "All ensembled checkpoints must share the same scheme."
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

    run_ensemble_inference(
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        checkpoint_dir=args.checkpoint_dir,
        checkpoints=args.checkpoints,
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

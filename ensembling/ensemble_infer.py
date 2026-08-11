import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.transforms import SpatialResample

try:
    from tqdm import tqdm
except ImportError:  # progress bars are a convenience, never a hard dependency
    tqdm = None


ARCH_ORDER = ["nnunet", "swin_unetr", "mavin"]
NUM_FOLDS = 5

ARCH_DIR_ALIASES = {
    "nnunet": {"nnunet", "nn_unet"},
    "swin_unetr": {"swin_unetr", "swinunetr", "swin"},
    "mavin": {"mavin", "ma_vin"},
}


MODALITY_ALIASES = {
    "t1c": ("t1c", "t1ce"),
    "t1n": ("t1n", "t1"),
    "t2f": ("t2f", "flair"),
    "t2w": ("t2w", "t2"),
}


MONAI_ORDER = ("t1c", "t1n", "t2f", "t2w")


REGION_DEFS = {
    "ET": (3,),
    "TC": (1, 3),
    "WT": (1, 2, 3),
}

FOLD_PATTERN = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)
NNUNET_CKPT_NAMES = ("checkpoint_final.pth", "checkpoint_best.pth")

AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}




class _NullProgressBar:
    """Stand-in used when progress is disabled or tqdm is not installed, so
    callers never have to branch on whether a bar exists."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix(self, **kwargs) -> None:
        pass

    def write(self, message: str) -> None:
        print(message)

    def close(self) -> None:
        pass


def make_progress_bar(total: int, desc: str, enabled: bool = True):
    
    if not enabled or tqdm is None:
        return _NullProgressBar()
    return tqdm(
        total=total, desc=desc, unit="case", dynamic_ncols=True,
        file=sys.stderr, smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )




def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Equal-weight, two-level soft-probability test-time ensemble over "
            "nnU-Net, SwinUNETR and MaViN (5 folds each)."
        )
    )
    p.add_argument("--models-dir", required=True,
                    help="Root dir with one sub-directory per architecture "
                         "(nnunet, swin_unetr, mavin), each with 5 fold checkpoints.")
    p.add_argument("--cases-dir", required=True,
                    help="Input cases directory (nested per-case subfolders or flat "
                         "layout) containing t1c/t1n/t2f/t2w NIfTI volumes.")
    p.add_argument("--output-dir", required=True,
                    help="Output directory for segmentations, optional probability "
                         "maps, and optional metrics CSV.")
    p.add_argument("--gt-dir", default=None,
                    help="Optional ground-truth directory. If given, per-case DSC "
                         "and HD95 are computed and written to metrics.csv. Skipped "
                         "silently if not given.")

    p.add_argument("--swin-model-config", default=None,
                    help="Path to SwinUNETR's model.yaml (architecture hyperparameters). "
                         "Default: <models-dir>/swin_unetr/configs/model.yaml (auto-derived "
                         "from the detected swin_unetr directory -- never resolved relative "
                         "to the current working directory).")
    p.add_argument("--mavin-model-config", default=None,
                    help="Path to MaViN's model.yaml (architecture hyperparameters). "
                         "Default: <models-dir>/mavin/configs/model.yaml (auto-derived "
                         "from the detected mavin directory).")
    p.add_argument("--mavin-repo-root", default=None,
                    help="Path to the MaViN source repo (for importing MambaVisionUNet). "
                         "Default: the detected <models-dir>/mavin directory itself -- "
                         "this project's mavin checkout keeps checkpoints and source "
                         "together, no separate mavin-hpc clone required.")

    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128],
                    help="Sliding window patch size for swin_unetr/mavin.")
    p.add_argument("--sw-batch-size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.7,
                    help="Sliding window overlap fraction, shared by all three architectures.")
    p.add_argument("--blend-mode", choices=["gaussian", "constant"], default="gaussian",
                    help="Sliding window blend mode for swin_unetr/mavin.")

    p.add_argument("--amp", action="store_true",
                    help="Enable autocast mixed precision for SwinUNETR/MaViN sliding-window "
                         "inference (memory savings). Default off (full fp32). Only takes "
                         "effect on --device cuda, matching swin_unetr/mavin-hpc training "
                         "convention. nnU-Net already runs mixed precision internally via its "
                         "own predictor, unaffected by this flag.")
    p.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16",
                    help="Autocast dtype when --amp is set (default bfloat16, matching "
                         "swin_unetr/mavin-hpc). Activation (sigmoid/softmax) is always "
                         "computed in float32 regardless of this setting.")

    p.add_argument("--nnunet-modality-order", nargs=4, default=None,
                    metavar=("CH0", "CH1", "CH2", "CH3"),
                    help="Modality order this specific nnU-Net checkpoint set's "
                         "imagesTr channels (_0000.._0003) were trained with, e.g. "
                         "'t1c t1n t2w t2f'. Different nnU-Net training runs in this "
                         "project have used different orders -- get this wrong and "
                         "channels get silently swapped. If omitted, it is inferred "
                         "from that run's own dataset.json channel_names; if that's "
                         "missing or ambiguous, the script aborts rather than guess.")
    p.add_argument("--nnunet-checkpoint-name", default=None,
                    help="Fixed checkpoint filename to use for every nnU-Net fold "
                         "(default: try checkpoint_final.pth, then checkpoint_best.pth).")
    p.add_argument("--nnunet-tile-step-size", type=float, default=None,
                    help="Overrides 1 - overlap for nnU-Net's own sliding window step size.")
    p.add_argument("--nnunet-use-mirroring", action="store_true",
                    help="Enable nnU-Net's flip-based test-time mirroring (default off, "
                         "to keep it a plain sliding-window pass like the other two "
                         "architectures).")

    p.add_argument("--weights-source", choices=["equal", "json"], default="equal",
                    help="How the 3 per-architecture probability maps are combined at "
                         "level 2. 'equal' (default) gives each architecture weight "
                         "1/3. 'json' loads frozen architecture weights produced by "
                         "scripts/oof_weights.py. Equal weighting stays the default "
                         "until the optimized weights are explicitly enabled.")
    p.add_argument("--weights-json", default=None,
                    help="Path to the frozen weights JSON. Required when "
                         "--weights-source json, ignored otherwise.")

    p.add_argument("--threshold", type=float, default=0.5,
                    help="Binarization threshold applied independently to each of the "
                         "3 region probabilities (ET, TC, WT) after level-2 combination, "
                         "before the priority-overwrite collapse to a discrete label "
                         "(see regions_to_classes()). Default 0.5.")

    p.add_argument("--keep-largest", action="store_true",
                    help="Keep only the largest connected foreground component in the "
                         "final segmentation (default off).")
    p.add_argument("--save-probs", action="store_true",
                    help="Also save the final (level-2) averaged region-probability map "
                         "(channels: ET, TC, WT) per case (default off).")
    p.add_argument("--save-arch-probs", action="store_true",
                    help="Also save each architecture's level-1 (fold-averaged) "
                         "region-probability map (channels: ET, TC, WT) "
                         "per case, for inspection (default off).")

    p.add_argument("--allow-incomplete", action="store_true",
                    help="Proceed even if fewer than 3 architectures or fewer than 5 "
                         "folds per architecture are found (averages over whatever is "
                         "found). Off by default -- incomplete layouts abort loudly.")
    p.add_argument("--limit", type=int, default=None,
                    help="Only process the first N cases (for smoke testing).")
    p.add_argument("--no-progress", action="store_true",
                    help="Disable the per-case progress bar and its ETA. The bar is "
                         "drawn on stderr and is a no-op if tqdm is not installed.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)



WEIGHT_SENSITIVE_SETTINGS = ("overlap", "roi", "blend_mode", "amp", "amp_dtype",
                              "nnunet_use_mirroring", "threshold")


def warn_on_settings_mismatch(fitted: dict, current: dict, path) -> None:


    mismatches = []
    for key in WEIGHT_SENSITIVE_SETTINGS:
        if key not in fitted or key not in current:
            continue
        was, now = fitted[key], current[key]
        if isinstance(was, list) or isinstance(now, list):
            was, now = list(was or []), list(now or [])
        if was != now:
            mismatches.append((key, was, now))
    if not mismatches:
        return
    print(f"WARNING: {path} was fitted under different inference settings than "
          f"this run. The learned weights may not be the best weights for the "
          f"probability maps this run produces:")
    for key, was, now in mismatches:
        print(f"  {key}: fitted with {was!r}, running with {now!r}")
    print("  Re-run scripts/oof_weights.py with matching settings, or accept the "
          "mismatch knowingly.")


def resolve_arch_weights(weights_source: str, weights_json: str | None,
                          active_archs: list, run_settings: dict | None = None) -> dict:


    if weights_source == "equal":
        return {arch: 1.0 / len(active_archs) for arch in active_archs}

    if not weights_json:
        raise RuntimeError("--weights-source json requires --weights-json PATH")
    path = Path(weights_json)
    if not path.is_file():
        raise RuntimeError(f"--weights-json path does not exist: {path}")

    import json  # noqa: PLC0415
    with open(path) as f:
        payload = json.load(f)
    if "weights" not in payload or not isinstance(payload["weights"], dict):
        raise RuntimeError(f"{path}: expected a top-level 'weights' object mapping "
                            f"architecture name to weight.")
    weights = {str(k): float(v) for k, v in payload["weights"].items()}

    missing = [a for a in active_archs if a not in weights]
    extra = [a for a in weights if a not in active_archs]
    if missing or extra:
        raise RuntimeError(
            f"{path}: weights cover {sorted(weights)} but the active architectures "
            f"are {sorted(active_archs)}"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
            + ". Architecture weights fitted on the full 3-architecture ensemble are "
              "not valid over a subset. Re-run scripts/oof_weights.py, or use "
              "--weights-source equal."
        )
    negative = {a: w for a, w in weights.items() if w < 0.0}
    if negative:
        raise RuntimeError(f"{path}: weights must be non-negative, got {negative}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError(f"{path}: weights must sum to 1, got {total:.8f} "
                            f"({weights}). Refusing to silently renormalize.")

    print(f"Loaded frozen architecture weights from {path}")
    for key in ("metric", "metric_value", "num_cases", "grid_step", "generated_utc"):
        if key in payload:
            print(f"  {key}: {payload[key]}")
    if run_settings is not None:
        warn_on_settings_mismatch(payload.get("settings", {}), run_settings, path)
    return weights




@dataclass
class ArchDetection:
    canonical: str
    dir: Path | None
    folds: dict  # fold_idx -> checkpoint Path
    fold_root: Path | None = None       # nnunet only: dir actually containing fold_X/
    plans_json: Path | None = None      # nnunet only
    dataset_json: Path | None = None    # nnunet only


def _normalize_dirname(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def find_arch_dir(models_dir: Path, canonical: str) -> Path | None:
    aliases = ARCH_DIR_ALIASES[canonical]
    for child in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if _normalize_dirname(child.name) in aliases:
            return child
    return None


def _extract_fold_idx(name: str) -> int | None:
    m = FOLD_PATTERN.search(name)
    return int(m.group(1)) if m else None


def find_generic_fold_checkpoints(
    arch_dir: Path, num_folds: int, extensions=(".pt", ".pth", ".ckpt")
) -> dict:
    candidates: dict = {}
    for path in arch_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        fold_idx = _extract_fold_idx(path.name)
        if fold_idx is None:
            fold_idx = _extract_fold_idx(path.parent.name)
        if fold_idx is None or not (0 <= fold_idx < num_folds):
            continue
        candidates.setdefault(fold_idx, []).append(path)

    result = {}
    for fold_idx, paths in candidates.items():
        if len(paths) > 1:
            paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(f"  WARNING: multiple checkpoint candidates for fold {fold_idx} "
                  f"in {arch_dir}; using most recent: {paths[0]}")
            print(f"           ignored: {[str(p) for p in paths[1:]]}")
        result[fold_idx] = paths[0]
    return result


def find_nnunet_fold_root(arch_dir: Path) -> Path:
    if any((arch_dir / f"fold_{i}").is_dir() for i in range(NUM_FOLDS)):
        return arch_dir
    for child in sorted(p for p in arch_dir.iterdir() if p.is_dir()):
        if any((child / f"fold_{i}").is_dir() for i in range(NUM_FOLDS)):
            return child
    return arch_dir


def find_nnunet_folds(arch_dir: Path, checkpoint_name: str | None):
    fold_root = find_nnunet_fold_root(arch_dir)
    result = {}
    names_to_try = [checkpoint_name] if checkpoint_name else list(NNUNET_CKPT_NAMES)
    for i in range(NUM_FOLDS):
        fold_dir = fold_root / f"fold_{i}"
        if not fold_dir.is_dir():
            continue
        ckpt = None
        for name in names_to_try:
            candidate = fold_dir / name
            if candidate.is_file():
                ckpt = candidate
                break
        if ckpt is None:
            pth_files = sorted(fold_dir.glob("*.pth"))
            ckpt = pth_files[0] if pth_files else None
        if ckpt is not None:
            result[i] = ckpt
    return result, fold_root, fold_root / "plans.json", fold_root / "dataset.json"


def detect_layout(models_dir: Path, nnunet_checkpoint_name: str | None) -> dict:
    result = {}
    for canonical in ARCH_ORDER:
        arch_dir = find_arch_dir(models_dir, canonical)
        if arch_dir is None:
            result[canonical] = ArchDetection(canonical, None, {})
            continue
        if canonical == "nnunet":
            folds, fold_root, plans_json, dataset_json = find_nnunet_folds(
                arch_dir, nnunet_checkpoint_name
            )
            result[canonical] = ArchDetection(
                canonical, arch_dir, folds, fold_root, plans_json, dataset_json
            )
        else:
            folds = find_generic_fold_checkpoints(arch_dir, NUM_FOLDS)
            result[canonical] = ArchDetection(canonical, arch_dir, folds)
    return result


def print_summary_table(detection: dict) -> None:
    print(f"{'Architecture':<14} {'Directory':<55} {'Folds found':<12}")
    print("-" * 90)
    for canonical in ARCH_ORDER:
        det = detection[canonical]
        dir_str = str(det.dir) if det.dir else "NOT FOUND"
        folds_str = f"{len(det.folds)}/{NUM_FOLDS}"
        print(f"{canonical:<14} {dir_str:<55} {folds_str:<12}")
        for i in range(NUM_FOLDS):
            if i in det.folds:
                print(f"    fold_{i}: {det.folds[i]}")
            else:
                print(f"    fold_{i}: MISSING")
        if canonical == "nnunet" and det.dir is not None:
            for label, path in (("plans.json", det.plans_json), ("dataset.json", det.dataset_json)):
                status = "found" if path and path.is_file() else "MISSING"
                print(f"    {label}: {path} ({status})")
    print("-" * 90)
    total_found = sum(len(detection[a].folds) for a in ARCH_ORDER)
    expected = len(ARCH_ORDER) * NUM_FOLDS
    print(f"Total: {total_found}/{expected} models found "
          f"({len(ARCH_ORDER)} architectures x {NUM_FOLDS} folds)")
    print("-" * 90)


def validate_detection(detection: dict, allow_incomplete: bool) -> bool:
    hard_fail = False
    for canonical in ARCH_ORDER:
        det = detection[canonical]
        n = len(det.folds)
        if det.dir is None:
            if allow_incomplete:
                print(f"WARNING: architecture '{canonical}' directory not found -- "
                      f"it will be SKIPPED entirely.")
            else:
                print(f"ERROR: could not find a directory for architecture "
                      f"'{canonical}' under --models-dir")
                hard_fail = True
        elif n != NUM_FOLDS:
            if allow_incomplete:
                print(f"WARNING: architecture '{canonical}' has {n}/{NUM_FOLDS} fold "
                      f"checkpoints -- averaging over {n} instead of {NUM_FOLDS}.")
            else:
                print(f"ERROR: architecture '{canonical}' has {n}/{NUM_FOLDS} fold "
                      f"checkpoints (expected {NUM_FOLDS}).")
                hard_fail = True
        if canonical == "nnunet" and det.dir is not None:
            for label, path in (("plans.json", det.plans_json), ("dataset.json", det.dataset_json)):
                if not (path and path.is_file()):
                    print(f"ERROR: nnU-Net {label} not found at {path}")
                    hard_fail = True
    if hard_fail and not allow_incomplete:
        print("Aborting. Fix the models directory, or pass --allow-incomplete to proceed anyway.")
        return False
    return True




def _strip_nifti_ext(name: str) -> str | None:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return None


def _match_modality_token(token: str) -> str | None:
    token = token.lower()
    for key, aliases in MODALITY_ALIASES.items():
        if token in aliases:
            return key
    return None


def _split_prefix_and_modality(stem: str):
    for delim in ("-", "_"):
        if delim in stem:
            prefix, _, last = stem.rpartition(delim)
            mod = _match_modality_token(last)
            if mod:
                return prefix, mod
    return stem, None


def detect_case_layout(cases_dir: Path) -> str:
    entries = sorted(cases_dir.iterdir())
    subdirs = [e for e in entries if e.is_dir()]
    nested_hits = 0
    for d in subdirs:
        if any(_strip_nifti_ext(f.name) is not None for f in d.iterdir() if f.is_file()):
            nested_hits += 1
    if nested_hits > 0:
        return "nested"

    flat_hits = any(_strip_nifti_ext(e.name) is not None for e in entries if e.is_file())
    if flat_hits:
        return "flat"

    raise RuntimeError(f"Could not detect a case layout under {cases_dir}: "
                        f"no NIfTI files found in per-case subfolders or directly inside it.")


def _match_modalities(files: list) -> dict:
    found = {}
    for path in files:
        stem = _strip_nifti_ext(path.name)
        if stem is None:
            continue
        _, mod = _split_prefix_and_modality(stem)
        if mod is None:
            continue
        if mod in found:
            raise RuntimeError(f"Duplicate modality '{mod}' match: {found[mod]} and {path}")
        found[mod] = path
    return found


def list_cases(cases_dir: Path, layout: str) -> list:
    cases = []
    if layout == "nested":
        for d in sorted(p for p in cases_dir.iterdir() if p.is_dir()):
            files = [f for f in d.iterdir() if f.is_file()]
            found = _match_modalities(files)
            missing = [m for m in MONAI_ORDER if m not in found]
            if missing:
                print(f"  WARNING: skipping case '{d.name}': missing modalities {missing}")
                continue
            cases.append({"case_id": d.name, "files": found})
    elif layout == "flat":
        groups: dict = {}
        for f in sorted(p for p in cases_dir.iterdir() if p.is_file()):
            stem = _strip_nifti_ext(f.name)
            if stem is None:
                continue
            prefix, mod = _split_prefix_and_modality(stem)
            if mod is None:
                continue
            groups.setdefault(prefix, {})[mod] = f
        for case_id, found in sorted(groups.items()):
            missing = [m for m in MONAI_ORDER if m not in found]
            if missing:
                print(f"  WARNING: skipping case '{case_id}': missing modalities {missing}")
                continue
            cases.append({"case_id": case_id, "files": found})
    else:
        raise RuntimeError(f"Unknown layout '{layout}'")
    return cases




def load_nifti(path: Path):
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    return data, img.affine.copy(), img.header.copy()


def load_case(files: dict):
    """Load each modality once; return {mod: array}, shared affine/header/shape."""
    mod_arrays = {}
    ref_affine = ref_shape = ref_header = None
    for mod, path in files.items():
        data, affine, header = load_nifti(path)
        if ref_affine is None:
            ref_affine, ref_shape, ref_header = affine, data.shape, header
        else:
            if data.shape != ref_shape:
                raise RuntimeError(
                    f"Modality '{mod}' ({path}) shape {data.shape} does not match "
                    f"reference shape {ref_shape} for the same case."
                )
            if not np.allclose(affine, ref_affine, atol=1e-3):
                raise RuntimeError(
                    f"Modality '{mod}' ({path}) affine does not match the reference "
                    f"affine for the same case. All modalities of a case must share "
                    f"one geometry."
                )
        mod_arrays[mod] = data
    return mod_arrays, ref_affine, ref_header, ref_shape


def stack_modalities(mod_arrays: dict, order: tuple) -> np.ndarray:
    return np.stack([mod_arrays[m] for m in order], axis=0)


def normalize_nonzero_channelwise(volume: np.ndarray) -> np.ndarray:
    """Match monai.transforms.NormalizeIntensityd(nonzero=True, channel_wise=True)."""
    out = np.zeros_like(volume, dtype=np.float32)
    for c in range(volume.shape[0]):
        ch = volume[c]
        mask = ch != 0
        if mask.any():
            mean = ch[mask].mean()
            std = ch[mask].std()
            std = std if std >= 1e-8 else 1e-8
            out[c] = np.where(mask, (ch - mean) / std, 0.0)
        else:
            out[c] = ch
    return out


def resample_prob(prob: np.ndarray, src_affine: np.ndarray, dst_affine: np.ndarray, dst_shape: tuple) -> np.ndarray:
    img = MetaTensor(torch.from_numpy(prob).float(), affine=torch.from_numpy(src_affine).float())
    resampler = SpatialResample()
    out = resampler(img=img, dst_affine=torch.from_numpy(dst_affine).float(),
                     spatial_size=tuple(dst_shape), mode="bilinear")
    return np.ascontiguousarray(np.asarray(out).astype(np.float32))


def assert_or_resample(prob: np.ndarray, affine: np.ndarray, ref_affine: np.ndarray,
                        ref_shape: tuple, case_id: str, tag: str) -> np.ndarray:
    shape_ok = tuple(prob.shape[1:]) == tuple(ref_shape)
    affine_ok = np.allclose(affine, ref_affine, atol=1e-2)
    if shape_ok and affine_ok:
        return prob
    print(f"  [{case_id}] {tag}: geometry mismatch (shape {prob.shape[1:]} vs "
          f"{ref_shape}, affine_ok={affine_ok}) -- resampling to original case geometry")
    resampled = resample_prob(prob, affine, ref_affine, ref_shape)
    if tuple(resampled.shape[1:]) != tuple(ref_shape):
        raise RuntimeError(
            f"[{case_id}] {tag}: resample failed to reach the original case geometry "
            f"(got {resampled.shape[1:]}, expected {ref_shape}). Refusing to average "
            f"misaligned probability maps."
        )
    return resampled




def build_swin_unetr(model_config: dict, device: torch.device):
    from monai.networks.nets import SwinUNETR
    mcfg = model_config.get("model", {})
    
    return SwinUNETR(
        in_channels=int(mcfg.get("in_channels", 4)),
        out_channels=int(mcfg.get("out_channels", 3)),
        feature_size=int(mcfg.get("feature_size", 48)),
        use_checkpoint=bool(mcfg.get("use_checkpoint", True)),
        dropout_path_rate=float(mcfg.get("dropout_path_rate", 0.0)),
        use_v2=bool(mcfg.get("use_v2", False)),
    ).to(device)


def _extract_mavin_depth(model_config: dict) -> int:
    depths = model_config.get("model", {}).get("encoder", {}).get("depths", 1)
    if isinstance(depths, list):
        return int(depths[0])
    return int(depths)


def build_mavin(model_config: dict, mavin_repo_root: Path, roi, device: torch.device):
    repo_str = str(mavin_repo_root.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    try:
        from src.mambaVisionUNet import MambaVisionUNet  # noqa: PLC0415
    except ModuleNotFoundError:
        from src.model.mambaVisionUNet import MambaVisionUNet  # noqa: PLC0415

    mcfg = model_config.get("model", {})
    return MambaVisionUNet(
        img_size=tuple(roi),
        in_channels=int(mcfg.get("in_channels", 4)),
        out_channels=int(mcfg.get("out_channels", 4)),
        feature_size=int(mcfg.get("feature_size", 48)),
        depths=_extract_mavin_depth(model_config),
        num_heads=int(mcfg.get("num_heads", 16)),
        d_state=int(mcfg.get("d_state", 16)),
        use_checkpoint=bool(mcfg.get("use_checkpoint", True)),
    ).to(device)


def load_checkpoint_into_model(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer_sliding_window(model, volume_norm: np.ndarray, roi, sw_batch_size: int,
                          overlap: float, mode: str, device: torch.device,
                          amp_enabled: bool = False,
                          amp_dtype: torch.dtype = torch.bfloat16,
                          activation: str = "softmax") -> np.ndarray:
    """activation: 'softmax' for class-based mutually-exclusive outputs
    (MaViN), 'sigmoid' for per-channel overlapping-region outputs
    (SwinUNETR's native [ET, TC, WT])."""
    tensor = torch.from_numpy(volume_norm).unsqueeze(0).float().to(device)
    amp_enabled = amp_enabled and device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=list(roi),
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode=mode,
        )
        if activation == "sigmoid":
            probs = torch.sigmoid(logits.float())
        elif activation == "softmax":
            probs = torch.softmax(logits.float(), dim=1)
        else:
            raise ValueError(f"Unknown activation '{activation}'")
    return probs.squeeze(0).cpu().numpy().astype(np.float32)




def load_nnunet_label_manager(plans_json: Path, dataset_json: Path):
    """Load nnU-Net's own LabelManager from plans.json/dataset.json, used only
    to check has_regions / regions_class_order -- never assumed."""
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: PLC0415

    with open(dataset_json) as f:
        dj = json.load(f)
    return PlansManager(str(plans_json)).get_label_manager(dj)


def _resolve_modality_token(raw: str) -> str | None:
    raw = raw.strip().lower()
    for canonical, aliases in MODALITY_ALIASES.items():
        if raw == canonical or raw in aliases:
            return canonical
    return None


def infer_nnunet_order_from_dataset_json(dataset_json: Path) -> tuple | None:
    
    if not dataset_json.is_file():
        return None
    try:
        with open(dataset_json) as f:
            dj = json.load(f)
        channel_names = dj.get("channel_names") or dj.get("modality")
        if not channel_names:
            return None
        indexed = sorted(channel_names.items(), key=lambda kv: int(kv[0]))
        resolved = [_resolve_modality_token(v) for _, v in indexed]
        if len(resolved) != 4 or None in resolved or len(set(resolved)) != 4:
            return None
        return tuple(resolved)
    except Exception:
        return None


def resolve_nnunet_order(dataset_json: Path, cli_order: list | None) -> tuple:
    
    inferred = infer_nnunet_order_from_dataset_json(dataset_json)
    if cli_order:
        resolved = [_resolve_modality_token(t) for t in cli_order]
        if None in resolved or len(set(resolved)) != 4:
            raise RuntimeError(
                f"--nnunet-modality-order {cli_order} is not a valid permutation "
                f"of t1c/t1n/t2f/t2w."
            )
        order = tuple(resolved)
        if inferred is not None and inferred != order:
            print(f"  WARNING: --nnunet-modality-order {order} disagrees with "
                  f"dataset.json's channel_names (inferred {inferred} from "
                  f"{dataset_json}). Using the explicit --nnunet-modality-order.")
        return order
    if inferred is not None:
        return inferred
    raise RuntimeError(
        f"Cannot determine this nnU-Net checkpoint's modality channel order: "
        f"{dataset_json} is missing, unreadable, or its channel_names don't "
        f"unambiguously resolve to t1c/t1n/t2f/t2w, and --nnunet-modality-order "
        f"was not given. Different nnU-Net training runs in this project have "
        f"used different orders -- guessing wrong silently swaps channels and "
        f"corrupts predictions. Pass e.g. --nnunet-modality-order t1c t1n t2w t2f "
        f"explicitly (in the order this checkpoint's imagesTr _0000.._0003 were "
        f"generated)."
    )


def class_probs_to_region_probs(probs: np.ndarray) -> np.ndarray:
    
    return np.stack(
        [probs[list(class_ids)].sum(axis=0) for class_ids in REGION_DEFS.values()],
        axis=0,
    )


def nnunet_probs_to_region_probs(probs: np.ndarray, label_manager) -> np.ndarray:
    
    if label_manager.has_regions:
        regions = label_manager.regions_class_order

        def _region_idx(target_classes):
            target = set(target_classes)
            for i, r in enumerate(regions):
                r_set = set(r) if hasattr(r, "__iter__") else {r}
                if r_set == target:
                    return i
            return None

        indices = [_region_idx(labels) for labels in REGION_DEFS.values()]
        if None in indices:
            raise RuntimeError(
                f"Cannot map BraTS regions to nnU-Net channels. "
                f"regions_class_order={regions}"
            )
        return np.stack([probs[i] for i in indices], axis=0)

    return class_probs_to_region_probs(probs)




class NNUnetFoldRunner:
    

    def __init__(self, fold_root: Path, fold_idx: int, checkpoint_path: Path,
                 device: torch.device, tile_step_size: float, use_gaussian: bool,
                 use_mirroring: bool, verbose: bool = False):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: PLC0415

        self.predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_device=(device.type == "cuda"),
            device=device,
            verbose=verbose,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        self.predictor.initialize_from_trained_model_folder(
            str(fold_root), use_folds=(fold_idx,), checkpoint_name=checkpoint_path.name,
        )

    def predict(self, volume_xyz: np.ndarray, spacing_xyz) -> np.ndarray:
        """volume_xyz: (4, X, Y, Z) raw (unnormalized) intensities, in the modality
        order resolved by resolve_nnunet_order() for this checkpoint set."""
        data_zyx = np.ascontiguousarray(np.transpose(volume_xyz, (0, 3, 2, 1)).astype(np.float32))
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
        props = {"spacing": spacing_zyx}
        _, probs_zyx = self.predictor.predict_single_npy_array(data_zyx, props, None, None, True)
        probs_xyz = np.transpose(probs_zyx, (0, 3, 2, 1))
        return np.ascontiguousarray(probs_xyz.astype(np.float32))




def regions_to_classes(preds: np.ndarray) -> np.ndarray:
    
    label = np.zeros(preds.shape[1:], dtype=np.uint8)
    label[preds[2]] = 2  # WT -> ED/SNFH
    label[preds[1]] = 1  # TC -> NCR
    label[preds[0]] = 3  # ET -> ET
    return label


def keep_largest_component(label: np.ndarray) -> np.ndarray:
    from scipy.ndimage import label as cc_label
    foreground = label > 0
    if not foreground.any():
        return label
    labeled, num = cc_label(foreground)
    if num <= 1:
        return label
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    out = label.copy()
    out[(labeled != largest) & foreground] = 0
    return out


def save_segmentation(label: np.ndarray, affine: np.ndarray, header, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(label.astype(np.uint8), affine, header=header)
    img.header.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))


def save_prob_map(prob: np.ndarray, affine: np.ndarray, header, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    moved = np.moveaxis(prob, 0, -1).astype(np.float32)
    img = nib.Nifti1Image(moved, affine, header=header)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(out_path))




def find_gt_file(gt_dir: Path, case_id: str) -> Path | None:
    candidates = []
    case_subdir = gt_dir / case_id
    if case_subdir.is_dir():
        candidates.extend(sorted(case_subdir.glob("*seg*.nii*")))
    candidates.extend(sorted(gt_dir.glob(f"{case_id}*seg*.nii*")))
    candidates.extend(sorted(gt_dir.glob(f"{case_id}.nii.gz")))
    for c in candidates:
        if c.is_file():
            return c
    return None


def compute_case_metrics(case_id: str, pred_label: np.ndarray, gt_path: Path, affine: np.ndarray) -> dict:
    gt_data, _, _ = load_nifti(gt_path)
    gt_label = gt_data.astype(np.uint8)
    if gt_label.shape != pred_label.shape:
        raise RuntimeError(
            f"[{case_id}] ground truth shape {gt_label.shape} does not match "
            f"prediction shape {pred_label.shape}"
        )
    spacing = nib.affines.voxel_sizes(affine).tolist()

    row = {"case_id": case_id}
    dices, hd95s = [], []
    for region, labels in REGION_DEFS.items():
        pred_mask = np.isin(pred_label, labels)
        gt_mask = np.isin(gt_label, labels)
        pred_t = torch.from_numpy(pred_mask).float()[None, None]
        gt_t = torch.from_numpy(gt_mask).float()[None, None]

        dice = compute_dice(y_pred=pred_t, y=gt_t, include_background=True, ignore_empty=True).item()
        if pred_mask.any() and gt_mask.any():
            hd = compute_hausdorff_distance(
                y_pred=pred_t, y=gt_t, include_background=True, percentile=95, spacing=spacing
            ).item()
        else:
            hd = float("nan")

        row[f"dice_{region.lower()}"] = dice
        row[f"hd95_{region.lower()}"] = hd
        dices.append(dice)
        hd95s.append(hd)

    row["dice_mean"] = float(np.nanmean(dices))
    row["hd95_mean"] = float(np.nanmean(hd95s))
    return row


def write_metrics_csv(rows: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    mean_row = {"case_id": "MEAN"}
    for key in fieldnames[1:]:
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
        mean_row[key] = float(np.mean(vals)) if vals else float("nan")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        writer.writerow(mean_row)




def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    models_dir = Path(args.models_dir)
    cases_dir = Path(args.cases_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("STEP 1: Directory detection (expect 3 architectures x 5 folds = 15 models)")
    print("=" * 90)
    detection = detect_layout(models_dir, args.nnunet_checkpoint_name)
    print_summary_table(detection)
    if not validate_detection(detection, args.allow_incomplete):
        sys.exit(1)

    print()
    print("=" * 90)
    print("STEP 2: Case discovery")
    print("=" * 90)
    layout = detect_case_layout(cases_dir)
    cases = list_cases(cases_dir, layout)
    print(f"Detected case layout: {layout}")
    print(f"Found {len(cases)} case(s) in {cases_dir}")
    if args.limit:
        cases = cases[: args.limit]
        print(f"Limiting to first {len(cases)} case(s) (--limit)")
    if not cases:
        print("No cases found, nothing to do.")
        return

    print()
    print("=" * 90)
    print("STEP 3: Loading models (all 15, once)")
    print("=" * 90)
    tile_step_size = (
        args.nnunet_tile_step_size if args.nnunet_tile_step_size is not None
        else max(0.01, 1.0 - args.overlap)
    )
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    amp_active = args.amp and device.type == "cuda"
    print(f"SwinUNETR/MaViN mixed precision: {'ON' if amp_active else 'off'}"
          + (f" (dtype={args.amp_dtype})" if amp_active else
             " (--amp requested but device is not cuda)" if args.amp else ""))

    swin_models, mavin_models, nnunet_runners = {}, {}, {}

    if detection["swin_unetr"].folds:
        swin_dir = detection["swin_unetr"].dir
        swin_cfg_path = args.swin_model_config or str(swin_dir / "configs" / "model.yaml")
        print(f"  SwinUNETR model config: {swin_cfg_path}")
        swin_cfg = load_yaml(swin_cfg_path)
        for fold_idx, ckpt_path in sorted(detection["swin_unetr"].folds.items()):
            print(f"  Loading swin_unetr fold {fold_idx}: {ckpt_path}")
            model = build_swin_unetr(swin_cfg, device)
            load_checkpoint_into_model(model, ckpt_path, device)
            swin_models[fold_idx] = model

    if detection["mavin"].folds:
        mavin_dir = detection["mavin"].dir
        mavin_repo_root = Path(args.mavin_repo_root) if args.mavin_repo_root else mavin_dir
        mavin_cfg_path = args.mavin_model_config or str(mavin_dir / "configs" / "model.yaml")
        print(f"  MaViN repo root: {mavin_repo_root}")
        print(f"  MaViN model config: {mavin_cfg_path}")
        mavin_cfg = load_yaml(mavin_cfg_path)
        for fold_idx, ckpt_path in sorted(detection["mavin"].folds.items()):
            print(f"  Loading mavin fold {fold_idx}: {ckpt_path}")
            model = build_mavin(mavin_cfg, mavin_repo_root, args.roi, device)
            load_checkpoint_into_model(model, ckpt_path, device)
            mavin_models[fold_idx] = model

    nn_det = detection["nnunet"]
    nn_label_manager = None
    nnunet_order = None
    if nn_det.folds:
        nn_label_manager = load_nnunet_label_manager(nn_det.plans_json, nn_det.dataset_json)
        print(f"  nnU-Net label scheme: "
              f"{'region-based' if nn_label_manager.has_regions else 'class-based'} "
              f"(has_regions={nn_label_manager.has_regions})")
        nnunet_order = resolve_nnunet_order(nn_det.dataset_json, args.nnunet_modality_order)
        print(f"  nnU-Net modality order (imagesTr _0000.._0003): {nnunet_order}")
        for fold_idx, ckpt_path in sorted(nn_det.folds.items()):
            print(f"  Loading nnunet fold {fold_idx}: {ckpt_path}")
            nnunet_runners[fold_idx] = NNUnetFoldRunner(
                fold_root=nn_det.fold_root,
                fold_idx=fold_idx,
                checkpoint_path=ckpt_path,
                device=device,
                tile_step_size=tile_step_size,
                use_gaussian=True,
                use_mirroring=args.nnunet_use_mirroring,
                verbose=False,
            )

    active_archs = [a for a in ARCH_ORDER if detection[a].folds]
    print(f"Active architectures for averaging: {active_archs}")

    run_settings = {
        "overlap": float(args.overlap),
        "roi": list(args.roi),
        "blend_mode": args.blend_mode,
        "amp": bool(args.amp),
        "amp_dtype": args.amp_dtype if args.amp else None,
        "nnunet_use_mirroring": bool(args.nnunet_use_mirroring),
    }
    arch_weights = resolve_arch_weights(args.weights_source, args.weights_json,
                                        active_archs, run_settings)
    print(f"Level-2 weighting ({args.weights_source}): "
          + ", ".join(f"{a}={arch_weights[a]:.4f}" for a in active_archs))

    metrics_rows = []

    print()
    print("=" * 90)
    print(f"STEP 4: Running inference on {len(cases)} case(s)")
    print("=" * 90)

    bar = make_progress_bar(len(cases), "Ensemble inference", enabled=not args.no_progress)

    for case in cases:
        case_id = case["case_id"]
        files = case["files"]
        t0 = time.time()

        mod_arrays, ref_affine, ref_header, ref_shape = load_case(files)
        monai_volume = stack_modalities(mod_arrays, MONAI_ORDER)
        spacing_xyz = nib.affines.voxel_sizes(ref_affine)

        final_sum, num_regions = None, None

        for arch in active_archs:
            fold_ids = sorted(detection[arch].folds.keys())
            fold_sum = None

            for fold_idx in fold_ids:
                if arch == "swin_unetr":
                    # Native region-sigmoid output [ET, TC, WT], used as-is.
                    norm_vol = normalize_nonzero_channelwise(monai_volume)
                    prob = infer_sliding_window(
                        swin_models[fold_idx], norm_vol, args.roi, args.sw_batch_size,
                        args.overlap, args.blend_mode, device,
                        amp_enabled=args.amp, amp_dtype=amp_dtype,
                        activation="sigmoid",
                    )
                elif arch == "mavin":
                    # Native class-based softmax output, converted into region
                    # space the same lossless way as nnU-Net's class-based branch.
                    norm_vol = normalize_nonzero_channelwise(monai_volume)
                    prob = infer_sliding_window(
                        mavin_models[fold_idx], norm_vol, args.roi, args.sw_batch_size,
                        args.overlap, args.blend_mode, device,
                        amp_enabled=args.amp, amp_dtype=amp_dtype,
                        activation="softmax",
                    )
                    prob = class_probs_to_region_probs(prob)
                elif arch == "nnunet":
                    nnunet_volume = stack_modalities(mod_arrays, nnunet_order)
                    prob = nnunet_runners[fold_idx].predict(nnunet_volume, spacing_xyz)
                    prob = nnunet_probs_to_region_probs(prob, nn_label_manager)
                else:
                    raise RuntimeError(f"Unknown architecture '{arch}'")

                prob = assert_or_resample(prob, ref_affine, ref_affine, ref_shape,
                                           case_id, tag=f"{arch} fold {fold_idx}")

                if num_regions is None:
                    num_regions = prob.shape[0]
                elif prob.shape[0] != num_regions:
                    raise RuntimeError(
                        f"[{case_id}] {arch} fold {fold_idx} produced {prob.shape[0]} "
                        f"channels, expected {num_regions} (mismatched region-probability "
                        f"conversion across architectures -- should always be 3: "
                        f"ET, TC, WT)."
                    )
                if fold_sum is None:
                    fold_sum = np.zeros((num_regions,) + tuple(ref_shape), dtype=np.float64)
                fold_sum += prob

            # Level 1 is always an equal average over the 5 folds: a test case was
            # in no training set, so every fold checkpoint is equally fair on it.
            arch_prob = (fold_sum / len(fold_ids)).astype(np.float32)

            if args.save_arch_probs:
                save_prob_map(arch_prob, ref_affine, ref_header,
                              output_dir / "arch_probs" / arch / f"{case_id}.nii.gz")

            # Level 2: architecture weights, equal by default. They already sum
            # to 1 (validated in resolve_arch_weights), so no divide follows.
            weighted = arch_prob.astype(np.float64) * arch_weights[arch]
            final_sum = weighted if final_sum is None else final_sum + weighted

        final_prob = final_sum.astype(np.float32)
        preds = final_prob >= args.threshold
        label = regions_to_classes(preds)

        if args.keep_largest:
            label = keep_largest_component(label)

        seg_path = output_dir / "segmentations" / f"{case_id}.nii.gz"
        save_segmentation(label, ref_affine, ref_header, seg_path)

        if args.save_probs:
            save_prob_map(final_prob, ref_affine, ref_header,
                          output_dir / "probabilities" / f"{case_id}.nii.gz")

        elapsed = time.time() - t0
        bar.write(f"  [{case_id}] done in {elapsed:.1f}s -> {seg_path}")

        if args.gt_dir:
            gt_path = find_gt_file(Path(args.gt_dir), case_id)
            if gt_path is not None:
                metrics_rows.append(compute_case_metrics(case_id, label, gt_path, ref_affine))
            else:
                bar.write(f"  [{case_id}] no ground truth found, skipping metrics for this case")

        bar.set_postfix(case=case_id)
        bar.update(1)

    bar.close()

    if args.gt_dir and metrics_rows:
        csv_path = output_dir / "metrics.csv"
        write_metrics_csv(metrics_rows, csv_path)
        print(f"Wrote metrics CSV: {csv_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

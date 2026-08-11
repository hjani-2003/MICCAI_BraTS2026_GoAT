"""
Usage
-----
    python scripts/oof_weights.py \
        --models-dir ./models --cases-dir /path/to/training --gt-dir /path/to/training \
        --weights-out outputs/oof_weights/ensemble_weights.json

Then apply the frozen weights at test time:

    python scripts/ensemble_infer.py ... \
        --weights-source json --weights-json outputs/oof_weights/ensemble_weights.json
"""

import argparse
import csv
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import ensemble_infer as ei  # noqa: E402



ARCH_ORDER = ei.ARCH_ORDER          # ["nnunet", "swin_unetr"]
NUM_FOLDS = ei.NUM_FOLDS            # 5
REGION_DEFS = ei.REGION_DEFS        # {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}
REGION_NAMES = list(REGION_DEFS.keys())


HD95_CAP_DEFAULT = 373.13


CACHE_FORMAT_VERSION = 3


CACHE_DTYPES = {"uint16": (np.uint16, 65535), "uint8": (np.uint8, 255)}


CACHE_BBOX_PAD = 2


SWEEP_CHUNK_ELEMS = 50_000_000




def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Out-of-fold simplex grid search for the 3 architecture-level ensemble "
            "weights (nnU-Net, SwinUNETR, MaViN). Writes a frozen weights JSON that "
            "ensemble_infer.py can load with --weights-source json."
        )
    )

    # -- paths -------------------------------------------------------------
    p.add_argument("--models-dir", required=True,
                   help="Root dir with one sub-directory per architecture (nnunet, "
                        "swin_unetr, mavin), each with 5 fold checkpoints.")
    p.add_argument("--cases-dir", required=True,
                   help="TRAINING cases directory (the cases the folds were built "
                        "from), nested per-case subfolders or flat layout.")
    p.add_argument("--gt-dir", required=True,
                   help="Ground-truth directory. May be the same path as --cases-dir "
                        "when segmentations live beside the modalities.")
    p.add_argument("--splits", default=None,
                   help="Path to a fold-assignment file. Accepts nnU-Net "
                        "splits_final.json / splits.json (list of {train, val}), the "
                        "BraTS json_list (training[].fold), or a flat {case_id: fold} "
                        "mapping. Auto-detected under the nnU-Net model dir if omitted.")
    p.add_argument("--weights-out", default="outputs/oof_weights/ensemble_weights.json",
                   help="Where to write the frozen weights JSON.")
    p.add_argument("--ranked-out", default=None,
                   help="Where to write the full ranked combo table as CSV. Default: "
                        "ranked_combos.csv beside --weights-out.")
    p.add_argument("--cache-dir", default=None,
                   help="Optional OOF probability cache directory. When set, each "
                        "case's 3 maps are stored cropped to the brain bounding box "
                        "and quantized, and a valid cache skips inference entirely. "
                        "Off by default: inference is fused with the grid search and "
                        "nothing is written to disk.")
    p.add_argument("--cache-dtype", choices=sorted(CACHE_DTYPES), default="uint16",
                   help="Cache quantization precision. uint16 (default, about 56 GB "
                        "for 1351 BraTS cases) perturbs region probabilities by ~2e-5 "
                        "and preserves the winning combo. uint8 (about 7 GB) perturbs "
                        "them by ~6e-3, which exceeds the gap between adjacent grid "
                        "points and can select a different winner. Ignored without "
                        "--cache-dir.")

    # -- search ------------------------------------------------------------
    p.add_argument("--grid-step", type=float, default=0.05,
                   help="Simplex grid resolution (default 0.05 -> 231 combinations).")
    p.add_argument("--empty-policy", choices=["brats", "ignore"], default="brats",
                   help="How a region with empty ground truth is scored. 'brats' "
                        "(default) matches the GoAT leaderboard: dropped only when the "
                        "prediction is ALSO empty; an empty-GT false positive scores "
                        "0.0 and is counted. 'ignore' drops every empty-GT region "
                        "unconditionally (monai compute_dice(ignore_empty=True)), "
                        "blind to empty-case over-segmentation -- the failure mode "
                        "that pulled an earlier ensemble below SwinUNETR alone.")
    p.add_argument("--metric", choices=["dice", "composite"], default="dice",
                   help="Selection metric. 'dice' is mean Dice over ET/TC/WT. "
                        "'composite' ranks the full grid by Dice, then rescores only "
                        "the top --top-k combos with mean_dice - alpha * capped HD95.")
    p.add_argument("--top-k", type=int, default=20,
                   help="How many Dice-ranked combos get rescored under --metric "
                        "composite (default 20). Ignored for --metric dice.")
    p.add_argument("--hd95-alpha", type=float, default=0.5,
                   help="Weight of the normalized HD95 penalty in the composite.")
    p.add_argument("--hd95-cap", type=float, default=HD95_CAP_DEFAULT,
                   help="HD95 values are capped at this distance (mm) and divided by "
                        "it, mapping them into [0, 1]. Default is the BraTS volume "
                        "diagonal.")
    p.add_argument("--report-top", type=int, default=20,
                   help="How many rows of the ranked table to print (the full table "
                        "always goes to CSV).")

    # -- model configs -----------------------------------------------------
    p.add_argument("--swin-model-config", default=None,
                   help="SwinUNETR model.yaml. Default <models-dir>/swin_unetr/configs/model.yaml")
    p.add_argument("--mavin-model-config", default=None,
                   help="MaViN model.yaml. Default <models-dir>/mavin/configs/model.yaml")
    p.add_argument("--mavin-repo-root", default=None,
                   help="MaViN source repo root. Default: the detected mavin dir.")

    # -- inference (kept identical to ensemble_infer.py defaults) ----------
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--sw-batch-size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.7,
                   help="Sliding window overlap fraction, shared by all architectures.")
    p.add_argument("--blend-mode", choices=["gaussian", "constant"], default="gaussian")
    p.add_argument("--amp", action="store_true",
                   help="Autocast mixed precision for SwinUNETR/MaViN (cuda only).")
    p.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--nnunet-modality-order", nargs=4, default=None,
                   metavar=("CH0", "CH1", "CH2", "CH3"),
                   help="Modality order this nnU-Net checkpoint set's imagesTr channels "
                        "(_0000.._0003) were trained with, e.g. 't1c t1n t2w t2f'. "
                        "Different nnU-Net training runs in this project have used "
                        "different orders -- get this wrong and channels get silently "
                        "swapped. If omitted, inferred from dataset.json channel_names; "
                        "if that's missing or ambiguous, the script aborts rather than "
                        "guess.")
    p.add_argument("--nnunet-checkpoint-name", default=None)
    p.add_argument("--nnunet-tile-step-size", type=float, default=None,
                   help="Overrides 1 - overlap for nnU-Net's sliding window step size.")
    p.add_argument("--nnunet-use-mirroring", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarization threshold applied independently to each of the 3 "
                        "region probabilities (ET, TC, WT) when scoring Dice/HD95 during "
                        "the fit. Must match what ensemble_infer.py's --threshold will use "
                        "at test time for the fitted weights to reflect the deployed "
                        "binarization (recorded in the weights JSON and cache fingerprint; "
                        "see ensemble_infer.py's WEIGHT_SENSITIVE_SETTINGS). Default 0.5.")

    # -- run control -------------------------------------------------------
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N assigned cases (smoke testing).")
    p.add_argument("--folds", type=int, nargs="+", default=None,
                   help="Restrict to these fold indices (smoke testing). Default all.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve the fold assignment and case list, print the plan, "
                        "and exit without loading models or running inference.")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable the per-case progress bar and its ETA. The bar is "
                        "drawn on stderr and is a no-op if tqdm is not installed.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p




def _case_id_from_image_entry(entry) -> str:
    """Derive the case id from a json_list 'image' entry the same way
    generate_nnunet_splits.py does: the parent directory name of the first
    modality path."""
    first = entry[0] if isinstance(entry, (list, tuple)) else entry
    parent = Path(first).parent.name
    if parent and parent != ".":
        return parent
    stem = ei._strip_nifti_ext(Path(first).name)
    return stem if stem else Path(first).name


def _parse_splits_payload(payload, source: Path) -> dict:
    """Return {case_id: fold_idx} from any of the three accepted layouts."""
    assignment: dict = {}

    def _assign(case_id: str, fold: int):
        if case_id in assignment and assignment[case_id] != fold:
            raise RuntimeError(
                f"{source}: case '{case_id}' is assigned to fold "
                f"{assignment[case_id]} and fold {fold}. A case must be held out by "
                f"exactly one fold."
            )
        assignment[case_id] = fold

    # Layout A: nnU-Net splits_final.json -- list of {"train": [...], "val": [...]}
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) \
            and "val" in payload[0]:
        for fold_idx, split in enumerate(payload):
            for case_id in split["val"]:
                _assign(str(case_id), fold_idx)

    # Layout B: BraTS json_list -- {"training": [{"fold": k, "image": [...]}, ...]}
    elif isinstance(payload, dict) and "training" in payload:
        for item in payload["training"]:
            if "fold" not in item:
                raise RuntimeError(
                    f"{source}: a training entry has no 'fold' key, so its held-out "
                    f"fold cannot be determined: {item}"
                )
            _assign(_case_id_from_image_entry(item["image"]), int(item["fold"]))

    # Layout C: flat {case_id: fold}
    elif isinstance(payload, dict) and all(
            isinstance(v, int) or (isinstance(v, str) and v.isdigit())
            for v in payload.values()):
        for case_id, fold in payload.items():
            _assign(str(case_id), int(fold))

    else:
        raise RuntimeError(
            f"{source}: unrecognized fold-assignment layout. Expected one of: "
            f"nnU-Net splits_final.json (list of dicts with a 'val' key), a BraTS "
            f"json_list (dict with a 'training' list whose items carry a 'fold' key), "
            f"or a flat {{case_id: fold}} mapping."
        )

    if not assignment:
        raise RuntimeError(f"{source}: parsed successfully but produced no case-to-fold "
                           f"assignments.")

    bad = {c: f for c, f in assignment.items() if not (0 <= f < NUM_FOLDS)}
    if bad:
        raise RuntimeError(
            f"{source}: fold indices must be in [0, {NUM_FOLDS - 1}]. Offenders: "
            f"{dict(list(bad.items())[:5])}"
        )
    return assignment


def find_splits_file(models_dir: Path, detection: dict, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise RuntimeError(f"--splits path does not exist: {path}")
        return path

    candidates: list[Path] = []
    nn = detection.get("nnunet")
    for root in [getattr(nn, "fold_root", None), getattr(nn, "dir", None), models_dir]:
        if root is None:
            continue
        for name in ("splits_final.json", "splits.json"):
            direct = Path(root) / name
            if direct.is_file():
                candidates.append(direct)
    if nn is not None and nn.dir is not None:
        for name in ("splits_final.json", "splits.json"):
            candidates.extend(sorted(Path(nn.dir).rglob(name)))

    for c in candidates:
        if c.is_file():
            print(f"Auto-detected fold assignment: {c}")
            return c

    raise RuntimeError(
        "Could not auto-detect a fold-assignment file. Looked for "
        "splits_final.json / splits.json under the nnU-Net model directory and "
        f"under {models_dir}. Pass one explicitly with --splits (nnU-Net "
        "splits_final.json, a BraTS json_list, or a {case_id: fold} mapping)."
    )


def load_fold_assignment(splits_path: Path) -> dict:
    with open(splits_path) as f:
        payload = json.load(f)
    assignment = _parse_splits_payload(payload, splits_path)
    counts = {k: 0 for k in range(NUM_FOLDS)}
    for fold in assignment.values():
        counts[fold] += 1
    print(f"Fold assignment: {len(assignment)} cases from {splits_path}")
    for k in range(NUM_FOLDS):
        print(f"  fold {k}: {counts[k]} held-out case(s)")
    return assignment


def bind_cases_to_folds(cases: list, assignment: dict) -> list:
    """Attach a fold index to every discovered case. Fail loudly on any case that
    cannot be assigned to exactly one fold."""
    bound, unassigned = [], []
    for case in cases:
        fold = assignment.get(case["case_id"])
        if fold is None:
            unassigned.append(case["case_id"])
            continue
        bound.append({**case, "fold": fold})

    if unassigned:
        shown = ", ".join(unassigned[:10])
        more = f" (and {len(unassigned) - 10} more)" if len(unassigned) > 10 else ""
        raise RuntimeError(
            f"{len(unassigned)} case(s) found under --cases-dir have no fold "
            f"assignment: {shown}{more}. OOF requires every case to be held out by "
            f"exactly one fold. Fix the splits file or point --cases-dir at the "
            f"training set the folds were built from."
        )

    found_ids = {c["case_id"] for c in cases}
    missing_on_disk = sorted(set(assignment) - found_ids)
    if missing_on_disk:
        shown = ", ".join(missing_on_disk[:10])
        more = f" (and {len(missing_on_disk) - 10} more)" if len(missing_on_disk) > 10 else ""
        print(f"WARNING: {len(missing_on_disk)} case(s) in the splits file were not "
              f"found under --cases-dir and will be skipped: {shown}{more}")
    return bound




def weight_grid(n_arch: int, step: float) -> list:
    """All length-n weight tuples on the simplex at the given step resolution.
    Built from integer counts so the weights are exact multiples of step and
    sum to exactly 1.0."""
    if not (0.0 < step <= 1.0):
        raise RuntimeError(f"--grid-step must be in (0, 1], got {step}")
    total = round(1.0 / step)
    if abs(total * step - 1.0) > 1e-9:
        raise RuntimeError(
            f"--grid-step {step} does not divide 1.0 evenly (1/step = {1.0 / step}). "
            f"Use a step like 0.05, 0.1 or 0.25."
        )

    combos: list = []

    def _recurse(remaining: int, budget: int, current: list):
        if remaining == 1:
            combos.append(tuple(current + [budget]))
            return
        for k in range(budget + 1):
            _recurse(remaining - 1, budget - k, current + [k])

    _recurse(n_arch, total, [])
    return [tuple(c / total for c in combo) for combo in combos]




def settings_fingerprint(args, detection: dict) -> str:
    
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "arch_order": ARCH_ORDER,
        "roi": list(args.roi),
        "sw_batch_size": args.sw_batch_size,
        "overlap": args.overlap,
        "blend_mode": args.blend_mode,
        "amp": bool(args.amp),
        "amp_dtype": args.amp_dtype if args.amp else None,
        "nnunet_use_mirroring": bool(args.nnunet_use_mirroring),
        "nnunet_tile_step_size": args.nnunet_tile_step_size,
        "nnunet_order": list(args.nnunet_order),
        "threshold": float(args.threshold),
        "checkpoints": {},
    }
    for arch in ARCH_ORDER:
        det = detection[arch]
        entries = {}
        for fold_idx, ckpt in sorted(det.folds.items()):
            stat = ckpt.stat()
            entries[str(fold_idx)] = [str(ckpt), int(stat.st_mtime), int(stat.st_size)]
        payload["checkpoints"][arch] = entries
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("ascii")
    return hashlib.sha1(blob).hexdigest()


def union_foreground_bbox(probs: np.ndarray, shape: tuple, pad: int, threshold: float) -> tuple:
    
    fg = (probs >= threshold).any(axis=(0, 1))
    if not fg.any():
        return (0, 1, 0, 1, 0, 1)
    lo = []
    hi = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        proj = fg.any(axis=other)
        idx = np.flatnonzero(proj)
        lo.append(max(0, int(idx[0]) - pad))
        hi.append(min(shape[axis], int(idx[-1]) + 1 + pad))
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def save_case_cache(path: Path, probs: np.ndarray, shape: tuple, fingerprint: str,
                    cache_dtype: str, threshold: float) -> None:
    
    np_dtype, max_value = CACHE_DTYPES[cache_dtype]
    path.parent.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1, z0, z1 = union_foreground_bbox(probs, shape, CACHE_BBOX_PAD, threshold)
    crop = probs[:, :, x0:x1, y0:y1, z0:z1]
    quant = np.rint(np.clip(crop, 0.0, 1.0) * max_value).astype(np_dtype)


    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(
            handle,
            fingerprint=np.array(fingerprint),
            orig_shape=np.array(shape, dtype=np.int64),
            bbox=np.array([x0, x1, y0, y1, z0, z1], dtype=np.int64),
            cache_dtype=np.array(cache_dtype),
            region_probs=quant,
        )
    tmp.replace(path)


def load_case_cache(path: Path, shape: tuple, fingerprint: str,
                    cache_dtype: str) -> np.ndarray | None:


    if not path.is_file():
        return None
    _, max_value = CACHE_DTYPES[cache_dtype]
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"]) != fingerprint:
                return None
            if tuple(int(v) for v in data["orig_shape"]) != tuple(shape):
                return None
            if str(data["cache_dtype"]) != cache_dtype:
                return None
            x0, x1, y0, y1, z0, z1 = (int(v) for v in data["bbox"])
            quant = data["region_probs"]
    except (OSError, ValueError, KeyError):
        return None

    probs = np.zeros((len(ARCH_ORDER), 3) + tuple(shape), dtype=np.float32)
    probs[:, :, x0:x1, y0:y1, z0:z1] = quant.astype(np.float32) / max_value
    return probs




class FoldModels:
    

    def __init__(self, fold_idx: int, detection: dict, args, device: torch.device,
                 log=print):
        self.fold_idx = fold_idx
        self.detection = detection
        self.args = args
        self.device = device
        self.log = log
        self._swin = None
        self._mavin = None
        self._nnunet = None
        self._nn_label_manager = None

    @property
    def swin(self):
        if self._swin is None:
            det = self.detection["swin_unetr"]
            cfg_path = self.args.swin_model_config or str(det.dir / "configs" / "model.yaml")
            ckpt = det.folds[self.fold_idx]
            self.log(f"    loading swin_unetr fold {self.fold_idx}: {ckpt}")
            model = ei.build_swin_unetr(ei.load_yaml(cfg_path), self.device)
            self._swin = ei.load_checkpoint_into_model(model, ckpt, self.device)
        return self._swin

    @property
    def mavin(self):
        if self._mavin is None:
            det = self.detection["mavin"]
            repo_root = Path(self.args.mavin_repo_root) if self.args.mavin_repo_root else det.dir
            cfg_path = self.args.mavin_model_config or str(det.dir / "configs" / "model.yaml")
            ckpt = det.folds[self.fold_idx]
            self.log(f"    loading mavin fold {self.fold_idx}: {ckpt}")
            model = ei.build_mavin(ei.load_yaml(cfg_path), repo_root, self.args.roi, self.device)
            self._mavin = ei.load_checkpoint_into_model(model, ckpt, self.device)
        return self._mavin

    @property
    def nnunet(self):
        if self._nnunet is None:
            det = self.detection["nnunet"]
            ckpt = det.folds[self.fold_idx]
            self.log(f"    loading nnunet fold {self.fold_idx}: {ckpt}")
            tile_step = (self.args.nnunet_tile_step_size
                         if self.args.nnunet_tile_step_size is not None
                         else max(0.01, 1.0 - self.args.overlap))
            self._nnunet = ei.NNUnetFoldRunner(
                fold_root=det.fold_root, fold_idx=self.fold_idx, checkpoint_path=ckpt,
                device=self.device, tile_step_size=tile_step, use_gaussian=True,
                use_mirroring=self.args.nnunet_use_mirroring, verbose=False,
            )
        return self._nnunet

    @property
    def nn_label_manager(self):
        if self._nn_label_manager is None:
            det = self.detection["nnunet"]
            self._nn_label_manager = ei.load_nnunet_label_manager(det.plans_json, det.dataset_json)
        return self._nn_label_manager

    def release(self):
        self._swin = self._mavin = self._nnunet = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def infer_case_oof(models: FoldModels, mod_arrays: dict, ref_affine, ref_shape,
                   case_id: str, args, amp_dtype) -> np.ndarray:
    
    monai_volume = ei.stack_modalities(mod_arrays, ei.MONAI_ORDER)
    norm_volume = ei.normalize_nonzero_channelwise(monai_volume)
    spacing_xyz = ei.nib.affines.voxel_sizes(ref_affine)

    out = np.zeros((len(ARCH_ORDER), 3) + tuple(ref_shape), dtype=np.float32)
    for a_idx, arch in enumerate(ARCH_ORDER):
        if arch == "swin_unetr":
            prob = ei.infer_sliding_window(
                models.swin, norm_volume, args.roi, args.sw_batch_size, args.overlap,
                args.blend_mode, models.device, amp_enabled=args.amp,
                amp_dtype=amp_dtype, activation="sigmoid")
        elif arch == "mavin":
            prob = ei.infer_sliding_window(
                models.mavin, norm_volume, args.roi, args.sw_batch_size, args.overlap,
                args.blend_mode, models.device, amp_enabled=args.amp,
                amp_dtype=amp_dtype, activation="softmax")
            prob = ei.class_probs_to_region_probs(prob)
        elif arch == "nnunet":
            nnunet_volume = ei.stack_modalities(mod_arrays, args.nnunet_order)
            prob = models.nnunet.predict(nnunet_volume, spacing_xyz)
            prob = ei.nnunet_probs_to_region_probs(prob, models.nn_label_manager)
        else:
            raise RuntimeError(f"Unknown architecture '{arch}'")


        prob = ei.assert_or_resample(prob, ref_affine, ref_affine, ref_shape,
                                     case_id, tag=f"{arch} fold {models.fold_idx}")
        if prob.shape != (3,) + tuple(ref_shape):
            raise RuntimeError(
                f"[{case_id}] {arch} fold {models.fold_idx} produced shape "
                f"{prob.shape}, expected {(3,) + tuple(ref_shape)}. Refusing to "
                f"weight misaligned probability maps."
            )
        out[a_idx] = prob
    return out


def iter_case_probs(bound_cases: list, detection: dict, args, device, amp_dtype,
                    fingerprint: str, cache_dir: Path | None, bar=None):



    log = bar.write if bar is not None else print
    gt_dir = Path(args.gt_dir)
    by_fold: dict = {}
    for case in bound_cases:
        by_fold.setdefault(case["fold"], []).append(case)

    for fold_idx in sorted(by_fold):
        fold_cases = by_fold[fold_idx]
        log(f"  fold {fold_idx}: {len(fold_cases)} held-out case(s)")
        models = FoldModels(fold_idx, detection, args, device, log=log)

        for case in fold_cases:
            case_id = case["case_id"]
            t0 = time.time()

            mod_arrays, ref_affine, _, ref_shape = ei.load_case(case["files"])

            gt_path = ei.find_gt_file(gt_dir, case_id)
            if gt_path is None:
                raise RuntimeError(
                    f"[{case_id}] no ground truth found under {gt_dir}. OOF weight "
                    f"optimization scores every case against ground truth, so a "
                    f"missing label cannot be skipped silently."
                )
            gt_data, gt_affine, _ = ei.load_nifti(gt_path)
            gt = gt_data.astype(np.uint8)
            if gt.shape != tuple(ref_shape):
                raise RuntimeError(
                    f"[{case_id}] ground truth shape {gt.shape} does not match the "
                    f"case geometry {tuple(ref_shape)} ({gt_path})."
                )
            if not np.allclose(gt_affine, ref_affine, atol=1e-2):
                raise RuntimeError(
                    f"[{case_id}] ground truth affine does not match the case affine "
                    f"({gt_path}). Refusing to score against a misaligned label."
                )

            probs = None
            cache_path = cache_dir / f"{case_id}.npz" if cache_dir else None
            if cache_path is not None:
                probs = load_case_cache(cache_path, ref_shape, fingerprint, args.cache_dtype)

            if probs is None:
                probs = infer_case_oof(models, mod_arrays, ref_affine, ref_shape,
                                       case_id, args, amp_dtype)
                if cache_path is not None:
                    save_case_cache(cache_path, probs, ref_shape, fingerprint,
                                     args.cache_dtype, args.threshold)
                source = "inferred"
            else:
                source = "cached"

            log(f"    [{case_id}] fold {fold_idx} {source} in {time.time() - t0:.1f}s")
            if bar is not None:
                bar.set_postfix(fold=fold_idx, src=source)
            yield case_id, probs, gt, ref_affine

        models.release()




def _region_mask(labels: torch.Tensor, class_ids) -> torch.Tensor:
    mask = labels == class_ids[0]
    for cid in class_ids[1:]:
        mask = mask | (labels == cid)
    return mask


class DiceAccumulator:
    

    def __init__(self, n_combos: int, device: torch.device):
        self.sum = torch.zeros((len(REGION_NAMES), n_combos), dtype=torch.float64, device=device)
        self.count = torch.zeros((len(REGION_NAMES), n_combos), dtype=torch.float64, device=device)
        self.n_cases = 0

    def update(self, dice: torch.Tensor, valid: torch.Tensor):
        self.sum += torch.where(valid, dice.double(), torch.zeros_like(dice, dtype=torch.float64))
        self.count += valid.double()
        self.n_cases += 1

    def per_region_mean(self) -> np.ndarray:
        counts = self.count.clamp(min=1.0)
        mean = (self.sum / counts).cpu().numpy()
        mean[self.count.cpu().numpy() == 0] = np.nan
        return mean

    def mean_dice(self) -> np.ndarray:
        return np.nanmean(self.per_region_mean(), axis=0)


def contested_view(probs_t: torch.Tensor, gt_t: torch.Tensor, threshold: float):
    
    contested = (probs_t >= threshold).any(dim=1).any(dim=0)   # (X, Y, Z)
    probs_c = probs_t[:, :, contested]                          # (2, 3, M)
    gt_c = gt_t[contested]                                      # (M,)
    gt_sums = torch.stack([
        _region_mask(gt_t, REGION_DEFS[r]).sum() for r in REGION_NAMES
    ]).double()                                                 # (3,) over the FULL volume
    gt_masks_c = torch.stack([
        _region_mask(gt_c, REGION_DEFS[r]) for r in REGION_NAMES
    ])                                                          # (3, M)
    return probs_c, gt_masks_c, gt_sums, contested


def score_case_dice(probs_c: torch.Tensor, gt_masks_c: torch.Tensor,
                    gt_sums: torch.Tensor, combos_t: torch.Tensor, threshold: float,
                    empty_policy: str):
    
    if empty_policy not in ("brats", "ignore"):
        raise RuntimeError(f"unknown empty_policy '{empty_policy}'")

    n_combos = combos_t.shape[0]
    n_regions = len(REGION_NAMES)
    m = probs_c.shape[2]

    dice = torch.zeros((n_regions, n_combos), dtype=torch.float32, device=probs_c.device)

    if m == 0:
        
        valid = (gt_sums > 0).unsqueeze(1).expand(n_regions, n_combos).clone()
        return dice, valid

    if empty_policy == "ignore":
        valid = (gt_sums > 0).unsqueeze(1).expand(n_regions, n_combos).clone()
    else:  # 'brats': filled per (region, combo) below (drop both-empty only)
        valid = torch.zeros((n_regions, n_combos), dtype=torch.bool, device=probs_c.device)

    chunk = max(1, min(n_combos, SWEEP_CHUNK_ELEMS // max(1, n_regions * m)))
    for start in range(0, n_combos, chunk):
        w = combos_t[start:start + chunk]                       # (c, n_arch)
        weighted = torch.einsum("ca,afm->cfm", w, probs_c)      # (c, n_regions, M)
        pred = weighted >= threshold                            # (c, n_regions, M)
        for r_idx in range(n_regions):
            p = pred[:, r_idx, :]                                # (c, M)
            inter = (p & gt_masks_c[r_idx].unsqueeze(0)).sum(dim=1).double()
            pred_sum = p.sum(dim=1).double()
            denom = pred_sum + gt_sums[r_idx]
            d = torch.where(denom > 0, 2.0 * inter / denom.clamp(min=1e-8),
                            torch.zeros_like(denom))
            dice[r_idx, start:start + chunk] = d.float()
            if empty_policy == "brats":
                # Drop only the both-empty combos (denom == 0); count everything
                # else, so empty-GT false positives (denom > 0, dice 0.0) hurt.
                valid[r_idx, start:start + chunk] = denom > 0
    return dice, valid


# ---------------------------------------------------------------------------
# HD95 (top-K rescoring only -- far too slow for the full grid)
# ---------------------------------------------------------------------------

def _hd95_region(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing) -> float:
    """HD95 in mm, computed on a padded union bounding box for speed. Returns NaN
    when the ground truth region is empty (nothing to measure against) and inf
    when the ground truth is non-empty but nothing was predicted."""
    if not gt_mask.any():
        return float("nan")
    if not pred_mask.any():
        return float("inf")

    union = pred_mask | gt_mask
    lo, hi = [], []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        idx = np.flatnonzero(union.any(axis=other))
        lo.append(max(0, int(idx[0]) - 1))
        hi.append(min(union.shape[axis], int(idx[-1]) + 2))
    sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))

    pred_t = torch.from_numpy(np.ascontiguousarray(pred_mask[sl])).float()[None, None]
    gt_t = torch.from_numpy(np.ascontiguousarray(gt_mask[sl])).float()[None, None]
    return ei.compute_hausdorff_distance(
        y_pred=pred_t, y=gt_t, include_background=True, percentile=95,
        spacing=list(spacing),
    ).item()


def score_case_composite(gt: np.ndarray, contested: torch.Tensor,
                         probs_c: torch.Tensor, combos_t: torch.Tensor, spacing,
                         threshold: float):


    n_combos = combos_t.shape[0]
    n_regions = len(REGION_NAMES)
    hd_raw = np.zeros((n_regions, n_combos), dtype=np.float64)
    valid = np.zeros((n_regions, n_combos), dtype=bool)

    contested_np = contested.cpu().numpy()
    gt_region_masks = {r: np.isin(gt, REGION_DEFS[r]) for r in REGION_NAMES}
    full_shape = gt.shape

    for c_idx in range(n_combos):
        w = combos_t[c_idx]
        if probs_c.shape[2] == 0:
            pred_full = {region: np.zeros(full_shape, dtype=bool) for region in REGION_NAMES}
        else:
            weighted = torch.einsum("a,afm->fm", w, probs_c)          # (n_regions, M)
            pred_c = (weighted >= threshold).cpu().numpy()            # (n_regions, M)
            pred_full = {}
            for r_idx, region in enumerate(REGION_NAMES):
                full = np.zeros(full_shape, dtype=bool)
                full[contested_np] = pred_c[r_idx]
                pred_full[region] = full

        for r_idx, region in enumerate(REGION_NAMES):
            gt_mask = gt_region_masks[region]
            if not gt_mask.any():
                continue
            pred_mask = pred_full[region]
            valid[r_idx, c_idx] = True
            
            hd_raw[r_idx, c_idx] = _hd95_region(pred_mask, gt_mask, spacing)
    return hd_raw, valid



def warn_on_best_checkpoints(detection: dict) -> None:


    offenders = []
    for arch in ARCH_ORDER:
        for fold_idx, ckpt in sorted(detection[arch].folds.items()):
            if "best" in ckpt.name.lower():
                offenders.append((arch, fold_idx, ckpt.name))
    if not offenders:
        return
    archs = sorted({a for a, _, _ in offenders})
    print()
    print(f"WARNING: {len(offenders)} checkpoint(s) across {archs} are 'best' "
          f"checkpoints, e.g. {offenders[0][2]}.")
    print("  A 'best' checkpoint is the epoch chosen by validation score on the fold")
    print("  it held out -- the same cases this search scores against. Early stopping")
    print("  used that split too. The OOF Dice is therefore optimistically biased by")
    print("  model selection, and the bias is not necessarily equal across the three")
    print("  architectures, so it can tilt the learned weights for a reason that is")
    print("  not about model quality.")
    print("  Prefer last-epoch checkpoints for ALL THREE architectures if you have")
    print("  them (nnU-Net checkpoint_final.pth via --nnunet-checkpoint-name;")
    print("  periodic model_epoch_N_fold_K.pth for swin_unetr/mavin). Mixing 'best'")
    print("  for one architecture and 'final' for another is worse than either.")
    print("  This does not invalidate the run. It is a bias to know about, not a bug.")


def rank_order(key: np.ndarray) -> np.ndarray:


    filled = np.where(np.isnan(key), -np.inf, key)
    return np.argsort(-filled, kind="stable")


def write_ranked_csv(path: Path, combos: list, mean_dice: np.ndarray,
                     per_region: np.ndarray, composite: np.ndarray | None,
                     hd_per_region: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = composite if composite is not None else mean_dice
    order = rank_order(key)

    fields = ["rank"] + [f"w_{a}" for a in ARCH_ORDER] + ["mean_dice"] + \
             [f"dice_{r.lower()}" for r in REGION_NAMES]
    if composite is not None:
        fields += ["composite"] + [f"hd95_norm_{r.lower()}" for r in REGION_NAMES]


    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rank, i in enumerate(order, start=1):
            row = {"rank": rank}
            for a_idx, arch in enumerate(ARCH_ORDER):
                row[f"w_{arch}"] = f"{combos[i][a_idx]:.4f}"
            row["mean_dice"] = f"{mean_dice[i]:.6f}"
            for r_idx, region in enumerate(REGION_NAMES):
                row[f"dice_{region.lower()}"] = f"{per_region[r_idx, i]:.6f}"
            if composite is not None:
                row["composite"] = ("" if np.isnan(composite[i]) else f"{composite[i]:.6f}")
                for r_idx, region in enumerate(REGION_NAMES):
                    v = hd_per_region[r_idx, i]
                    row[f"hd95_norm_{region.lower()}"] = ("" if np.isnan(v) else f"{v:.6f}")
            writer.writerow(row)


def print_ranked_table(combos: list, mean_dice: np.ndarray, composite: np.ndarray | None,
                       top_n: int) -> None:
    key = composite if composite is not None else mean_dice
    order = rank_order(key)
    header = f"{'rank':<6}" + "".join(f"{'w_' + a:<14}" for a in ARCH_ORDER) + f"{'mean_dice':<12}"
    if composite is not None:
        header += f"{'composite':<12}"
    print(header)
    print("-" * len(header))
    for rank, i in enumerate(order[:top_n], start=1):
        line = f"{rank:<6}" + "".join(f"{combos[i][a]:<14.2f}" for a in range(len(ARCH_ORDER)))
        line += f"{mean_dice[i]:<12.6f}"
        if composite is not None:
            line += ("{:<12}".format("n/a") if np.isnan(composite[i])
                     else f"{composite[i]:<12.6f}")
        print(line)
    print("-" * len(header))


def write_weights_json(path: Path, combo: tuple, metric_name: str, metric_value: float,
                       mean_dice_value: float, args, n_cases: int, n_combos: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "arch_order": list(ARCH_ORDER),
        "weights": {arch: round(float(combo[i]), 6) for i, arch in enumerate(ARCH_ORDER)},
        "metric": metric_name,
        "metric_value": round(float(metric_value), 6),
        "mean_dice": round(float(mean_dice_value), 6),
        "empty_policy": args.empty_policy,
        "num_cases": int(n_cases),
        "num_combos": int(n_combos),
        "grid_step": float(args.grid_step),
        "settings": {
            "overlap": float(args.overlap),
            "roi": list(args.roi),
            "blend_mode": args.blend_mode,
            "amp": bool(args.amp),
            "amp_dtype": args.amp_dtype if args.amp else None,
            "nnunet_use_mirroring": bool(args.nnunet_use_mirroring),
            "threshold": float(args.threshold),
            "empty_policy": args.empty_policy,
            "hd95_alpha": float(args.hd95_alpha) if metric_name == "composite" else None,
            "hd95_cap": float(args.hd95_cap) if metric_name == "composite" else None,
            "top_k": int(args.top_k) if metric_name == "composite" else None,
        },
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Architecture weights fitted on single-fold out-of-fold predictions. "
            "At test time they are applied to 5-fold averaged per-architecture maps. "
            "See the honest-seam section of scripts/oof_weights.py."
        ),
    }
    with open(path, "w", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=False)
        f.write("\n")




def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    models_dir = Path(args.models_dir)
    cases_dir = Path(args.cases_dir)
    weights_out = Path(args.weights_out)
    ranked_out = Path(args.ranked_out) if args.ranked_out else weights_out.parent / "ranked_combos.csv"
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    print("=" * 90)
    print("STEP 0: Directory detection (3 architectures x 5 folds)")
    print("=" * 90)
    detection = ei.detect_layout(models_dir, args.nnunet_checkpoint_name)
    ei.print_summary_table(detection)
    
    if not ei.validate_detection(detection, allow_incomplete=False):
        sys.exit(1)
    warn_on_best_checkpoints(detection)


    nnunet_order = ei.resolve_nnunet_order(
        detection["nnunet"].dataset_json, args.nnunet_modality_order)
    print(f"nnU-Net modality order (imagesTr _0000.._0003): {nnunet_order}")
    args.nnunet_order = nnunet_order

    print()
    print("=" * 90)
    print("STEP 1: Fold assignment and case discovery")
    print("=" * 90)
    splits_path = find_splits_file(models_dir, detection, args.splits)
    assignment = load_fold_assignment(splits_path)

    layout = ei.detect_case_layout(cases_dir)
    cases = ei.list_cases(cases_dir, layout)
    print(f"Detected case layout: {layout}")
    print(f"Found {len(cases)} case(s) under {cases_dir}")

    bound_cases = bind_cases_to_folds(cases, assignment)
    if args.folds is not None:
        keep = set(args.folds)
        bound_cases = [c for c in bound_cases if c["fold"] in keep]
        print(f"Restricted to fold(s) {sorted(keep)}: {len(bound_cases)} case(s)")
    if args.limit:
        bound_cases = bound_cases[: args.limit]
        print(f"Limiting to first {len(bound_cases)} case(s) (--limit)")
    if not bound_cases:
        print("No cases to score, nothing to do.")
        return

    combos = weight_grid(len(ARCH_ORDER), args.grid_step)
    print(f"Simplex grid: step={args.grid_step} -> {len(combos)} weight combination(s)")
    print(f"Empty-region policy: {args.empty_policy}"
          + ("  (leaderboard convention: empty-GT false positives score 0.0)"
             if args.empty_policy == "brats" else "  (drops empty-GT regions)"))
    print(f"Metric: {args.metric}"
          + (f" (top-{args.top_k} rescored, alpha={args.hd95_alpha}, cap={args.hd95_cap})"
             if args.metric == "composite" else ""))
    if args.amp and device.type == "cuda":
        print("WARNING: --amp is enabled for a WEIGHT SEARCH. Autocast applies only to "
              "SwinUNETR and MaViN; nnU-Net always runs its own internal autocast, so "
              "--amp penalizes exactly 2 of the 3 architectures being weighted. That is "
              "a systematic bias, not noise: it does not average out over cases, and it "
              "shifts weight toward nnU-Net for a floating-point reason rather than a "
              "modelling one. Measured, bfloat16 changed the winning combo in 7 of 10 "
              "synthetic cases. Prefer full float32 here and use --cache-dir to make "
              "re-runs cheap.")

    if cache_dir is None:
        print("Cache: disabled (fused inference, nothing written to disk)")
        if args.metric == "composite":
            print("  NOTE: --metric composite makes a second pass over every case to "
                  "score HD95 on the top-K combos. Without --cache-dir that second "
                  "pass re-runs inference, roughly doubling the total runtime.")
    else:
        print(f"Cache: {cache_dir} (dtype={args.cache_dtype})")
        if args.cache_dtype == "uint8":
            print("  WARNING: uint8 quantization perturbs region probabilities by about "
                  "6e-3, which is larger than the gap between adjacent grid points. The "
                  "combo this run selects may differ from the one an uncached (exact) run "
                  "would select. Use --cache-dtype uint16 if the winner must be exact.")

    if args.dry_run:
        print("\n--dry-run: plan resolved, exiting before any model is loaded.")
        return

    fingerprint = settings_fingerprint(args, detection)
    amp_dtype = ei.AMP_DTYPES[args.amp_dtype]
    combos_t = torch.tensor(combos, dtype=torch.float32, device=device)

    print()
    print("=" * 90)
    print(f"STEP 2: Fused OOF inference + Dice sweep over {len(bound_cases)} case(s)")
    print("=" * 90)

    acc = DiceAccumulator(len(combos), device)
    bar = ei.make_progress_bar(len(bound_cases), "OOF inference + Dice sweep",
                               enabled=not args.no_progress)
    for case_id, probs, gt, _ in iter_case_probs(
            bound_cases, detection, args, device, amp_dtype, fingerprint, cache_dir, bar):
        probs_t = torch.from_numpy(probs).to(device)
        gt_t = torch.from_numpy(gt.astype(np.int16)).to(device)
        probs_c, gt_masks_c, gt_sums, _ = contested_view(probs_t, gt_t, args.threshold)
        dice, valid = score_case_dice(probs_c, gt_masks_c, gt_sums, combos_t,
                                      args.threshold, args.empty_policy)
        acc.update(dice, valid)
        del probs_t, gt_t, probs_c, gt_masks_c
        bar.update(1)
    bar.close()

    if acc.n_cases == 0:
        print("No cases were scored.")
        sys.exit(1)

    per_region = acc.per_region_mean()
    mean_dice = acc.mean_dice()

    composite = hd_per_region = None
    if args.metric == "composite":
        print()
        print("=" * 90)
        print(f"STEP 3: HD95 rescoring of the top {args.top_k} Dice-ranked combos")
        print("=" * 90)
        top_k = min(args.top_k, len(combos))
        top_idx = rank_order(mean_dice)[:top_k]
        top_combos_t = combos_t[torch.from_numpy(np.ascontiguousarray(top_idx)).to(device)]

        hd_sum = np.zeros((len(REGION_NAMES), top_k), dtype=np.float64)
        hd_cnt = np.zeros((len(REGION_NAMES), top_k), dtype=np.float64)
        bar = ei.make_progress_bar(len(bound_cases), f"HD95 rescore (top-{top_k})",
                                   enabled=not args.no_progress)
        for case_id, probs, gt, ref_affine in iter_case_probs(
                bound_cases, detection, args, device, amp_dtype, fingerprint, cache_dir, bar):
            probs_t = torch.from_numpy(probs).to(device)
            gt_t = torch.from_numpy(gt.astype(np.int16)).to(device)
            probs_c, _, _, contested = contested_view(probs_t, gt_t, args.threshold)
            spacing = ei.nib.affines.voxel_sizes(ref_affine).tolist()
            hd_raw, valid = score_case_composite(gt, contested, probs_c,
                                                 top_combos_t, spacing, args.threshold)
            # +inf (non-empty GT, empty prediction) caps to hd95_cap, normalizing
            # to the maximum penalty of 1.0. Invalid entries are masked to 0 and
            # never counted, so they cannot dilute the mean.
            capped = np.minimum(hd_raw, args.hd95_cap) / args.hd95_cap
            hd_sum += np.where(valid, capped, 0.0)
            hd_cnt += valid.astype(np.float64)
            del probs_t, gt_t, probs_c
            bar.update(1)
        bar.close()

        with np.errstate(invalid="ignore", divide="ignore"):
            hd_mean_top = np.where(hd_cnt > 0, hd_sum / np.clip(hd_cnt, 1, None), np.nan)

        hd_per_region = np.full((len(REGION_NAMES), len(combos)), np.nan)
        hd_per_region[:, top_idx] = hd_mean_top
        composite = np.full(len(combos), np.nan)
        composite[top_idx] = mean_dice[top_idx] - args.hd95_alpha * np.nanmean(hd_mean_top, axis=0)

    print()
    print("=" * 90)
    print("STEP 4: Ranked results")
    print("=" * 90)
    print_ranked_table(combos, mean_dice, composite, args.report_top)


    key = composite if composite is not None else mean_dice
    best = int(rank_order(key)[0])
    best_combo = combos[best]

    print()
    print("Winning architecture weights (OOF):")
    for a_idx, arch in enumerate(ARCH_ORDER):
        print(f"  w_{arch:<12} = {best_combo[a_idx]:.2f}")
    print(f"  mean_dice     = {mean_dice[best]:.6f}")
    if composite is not None:
        print(f"  composite     = {composite[best]:.6f}")
    print(f"  cases scored  = {acc.n_cases}")

    write_ranked_csv(ranked_out, combos, mean_dice, per_region, composite, hd_per_region)
    write_weights_json(weights_out, best_combo, args.metric, float(key[best]),
                       float(mean_dice[best]), args, acc.n_cases, len(combos))
    print()
    print(f"Wrote ranked table:   {ranked_out}")
    print(f"Wrote frozen weights: {weights_out}")
    print()
    print("Equal weights remain the default at test time. To use these weights, run "
          "ensemble_infer.py with:")
    print(f"  --weights-source json --weights-json {weights_out}")
    print()
    print("Reminder: these weights were fitted on SINGLE-FOLD out-of-fold predictions "
          "and are applied to 5-FOLD AVERAGED maps at test time. That transfer is an "
          "assumption, not an identity. See the honest-seam section of this module.")


if __name__ == "__main__":
    main()

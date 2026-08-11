#!/usr/bin/env python3
"""The submitted two-stage inference-time post-processing pipeline.

    stage 1  empty-mask fallback   substitute the donor on subjects predicted empty
    stage 2  ET size filter        drop ET components < T voxels, relabel to NCR

Donor first, filter last. The order is load-bearing: filter-first empties ET on subjects
whose components are all sub-threshold, the donor then re-fills it, and the filter's
largest win is undone on exactly the subjects it was meant to fix. Input predictions are
never modified.

    python run_postprocessing.py \\
        --pred-dir  /path/to/nnunet_predictions \\
        --donor-dir /path/to/hrnet_predictions \\
        --out-dir   submission_final \\
        --min-et-voxels 75 --zip

Omitting --donor-dir runs the size filter alone. Labels: 0=background, 1=NCR, 2=edema,
3=ET.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from empty_mask_fallback import apply_fallback, is_empty
from et_sizefilter import filter_et


def regions_of(seg: np.ndarray) -> tuple[int, int, int]:
    """(ET, TC, WT) voxel counts."""
    return int((seg == 3).sum()), int(((seg == 1) | (seg == 3)).sum()), int((seg > 0).sum())


def load_seg(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="Primary predictions (flat <case_id>.nii.gz per subject)")
    ap.add_argument("--donor-dir", type=Path, default=None,
                    help="Donor predictions, same filenames. Omit to run the size filter alone.")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Destination for post-processed predictions (created if absent)")
    ap.add_argument("--min-et-voxels", type=int, default=75,
                    help="Drop ET components smaller than this. 0 disables. Default 75.")
    ap.add_argument("--zip", action="store_true", help="Also write <out-dir>.zip for upload")
    ap.add_argument("--report", type=Path, default=None,
                    help="Write a per-subject JSON record of every action taken")
    args = ap.parse_args()

    files = sorted(args.pred_dir.glob("*.nii.gz"))
    if not files:
        print(f"no .nii.gz found under {args.pred_dir}", file=sys.stderr)
        return 1
    if args.donor_dir is None:
        print("no --donor-dir given: running the ET size filter only (stage 2 of 2)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_fallback = n_donor_empty = n_donor_missing = n_filtered = n_et_emptied = 0

    for i, f in enumerate(files, 1):
        img = nib.load(str(f))
        seg = np.asanyarray(img.dataobj).astype(np.uint8)
        row: dict[str, object] = {"subject": f.name[: -len(".nii.gz")]}

        if is_empty(seg) and args.donor_dir is not None:
            donor_path = args.donor_dir / f.name
            if not donor_path.exists():
                n_donor_missing += 1
                row["fallback"] = "donor file missing"
                print(f"  WARNING: {f.name} is empty and has no donor file", file=sys.stderr)
            else:
                seg, substituted = apply_fallback(seg, load_seg(donor_path))
                if substituted:
                    n_fallback += 1
                    row["fallback"] = "substituted"
                    row["donor_regions"] = regions_of(seg)
                    print(f"  fallback: {f.name} <- donor ({regions_of(seg)[2]} WT voxels)")
                else:
                    n_donor_empty += 1
                    row["fallback"] = "donor also empty"
                    print(f"  {f.name}: empty, but the donor is empty too -- left as-is")
        elif is_empty(seg):
            row["fallback"] = "empty, no donor supplied"

        if args.min_et_voxels > 0:
            before_et, before_tc, before_wt = regions_of(seg)
            seg, removed = filter_et(seg, args.min_et_voxels)
            if removed:
                after_et, after_tc, after_wt = regions_of(seg)
                assert (after_tc, after_wt) == (before_tc, before_wt), (
                    f"{f.name}: TC/WT changed ({before_tc},{before_wt}) -> "
                    f"({after_tc},{after_wt}) -- aborting rather than writing a corrupted mask"
                )
                n_filtered += 1
                row["et_voxels_removed"] = removed
                emptied = before_et > 0 and after_et == 0
                if emptied:
                    n_et_emptied += 1
                    row["et_emptied"] = True
                print(f"  filter  : {f.name} removed {removed} ET voxel(s)"
                      + (" (ET emptied entirely)" if emptied else ""))

        out = nib.Nifti1Image(seg.astype(np.uint8), affine=img.affine, header=img.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, str(args.out_dir / f.name))
        rows.append(row)

        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    print(f"\nsubjects processed          : {len(files)}")
    print(f"empty-mask fallbacks applied: {n_fallback}")
    if n_donor_empty:
        print(f"  empty, donor empty too    : {n_donor_empty}")
    if n_donor_missing:
        print(f"  empty, donor file missing : {n_donor_missing}")
    print(f"ET components removed in    : {n_filtered}")
    print(f"  ET emptied entirely       : {n_et_emptied}")
    print(f"written to                  : {args.out_dir}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rows, indent=2))
        print(f"per-subject report          : {args.report}")
    if args.zip:
        shutil.make_archive(str(args.out_dir), "zip", root_dir=args.out_dir)
        print(f"zipped to                   : {args.out_dir}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

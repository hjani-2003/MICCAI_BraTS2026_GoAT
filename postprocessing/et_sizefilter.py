#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def filter_et(seg: np.ndarray, min_et_voxels: int) -> tuple[np.ndarray, int]:
    """Relabel ET components smaller than min_et_voxels to NCR, leaving TC and WT intact.

    Returns (segmentation, voxels relabelled). Uses scipy's default 6-connectivity, which
    is what T=75 was calibrated against.
    """
    et = seg == 3
    if not et.any():
        return seg, 0
    lab, n = ndimage.label(et)
    if n == 0:
        return seg, 0
    sizes = np.bincount(lab.ravel())
    small = np.isin(lab, np.flatnonzero(sizes < min_et_voxels)) & et
    if not small.any():
        return seg, 0
    out = seg.copy()
    out[small] = 1
    return out, int(small.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Directory of predictions to filter IN PLACE (flat .nii.gz per case)")
    ap.add_argument("--min-et-voxels", type=int, default=75,
                    help="Drop ET components smaller than this. 0 disables. Default 75.")
    args = ap.parse_args()

    cases = sorted(args.output_dir.glob("*.nii.gz"))
    if not cases:
        print(f"[et_sizefilter] no predictions found under {args.output_dir}", file=sys.stderr)
        return 0
    if args.min_et_voxels <= 0:
        print(f"[et_sizefilter] --min-et-voxels={args.min_et_voxels}: disabled, "
              f"skipping all {len(cases)} case(s)")
        return 0

    n_filtered = n_emptied = 0
    for p in cases:
        img = nib.load(str(p))
        seg = np.asanyarray(img.dataobj).astype(np.uint8)
        had_et = (seg == 3).any()

        new_seg, removed = filter_et(seg, args.min_et_voxels)
        if not removed:
            continue

        n_filtered += 1
        emptied = had_et and not (new_seg == 3).any()
        n_emptied += emptied

        out = nib.Nifti1Image(new_seg, affine=img.affine, header=img.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, str(p))
        print(f"[et_sizefilter] {p.name}: removed {removed} ET voxel(s)"
              + (" (ET emptied entirely -> relabelled to NCR)" if emptied else ""))

    print(f"[et_sizefilter] done: {len(cases)} case(s) checked, {n_filtered} had ET "
          f"component(s) removed ({n_emptied} emptied entirely)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

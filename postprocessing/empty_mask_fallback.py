#!/usr/bin/env python3

from __future__ import annotations

import numpy as np


def is_empty(seg: np.ndarray) -> bool:
    return not (seg > 0).any()


def apply_fallback(seg: np.ndarray, donor_seg: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """Substitute donor_seg when seg is empty and the donor is not.

    Returns (segmentation to write, whether the substitution happened). An empty donor is
    declined: it would change nothing, and counting it would overstate how often the rule
    actually fires.
    """
    if donor_seg is None or not is_empty(seg) or is_empty(donor_seg):
        return seg, False
    return donor_seg, True

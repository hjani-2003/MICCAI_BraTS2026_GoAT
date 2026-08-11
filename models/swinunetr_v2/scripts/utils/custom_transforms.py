from monai.transforms import (
    MapTransform,
)
import torch


class ConvertToBraTSRegionsd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        for key in self.keys:
            seg = data[key].clone()
            if seg.shape[0] == 1:
                seg = seg[0]
            
            # ET: Enhancing Tumor (label 3)
            mask_ET = (seg == 3)
            # TC: Tumor Core (labels 1 and 3)
            mask_TC = (seg == 1) | (seg == 3)
            # WT: Whole Tumor (labels 1, 2, and 3)
            mask_WT = (seg == 1) | (seg == 2) | (seg == 3)
            
            data[key] = torch.stack([mask_ET, mask_TC, mask_WT], dim=0).float()

        return data


class ConvertToMultiClassd(MapTransform):
    """Keep the segmentation as a single-channel integer map with mutually-exclusive
    classes for softmax training:
        0 = background, 1 = NCR, 2 = ED, 3 = ET.

    BraTS-GoAT rasters use 1=NCR, 2=ED, 3=ET already, so this is mostly an
    identity that (a) guarantees a leading channel dim of size 1 and (b) remaps a
    stray ET label 4 -> 3 for robustness across BraTS conventions.
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            seg = data[key].clone()
            if seg.shape[0] == 1:
                seg = seg[0]

            # Remap so ET is class 3 regardless of source convention (3 or 4).
            seg[seg == 4] = 3

            # (1, H, W, D) integer class map for to_onehot_y / softmax loss.
            data[key] = seg.unsqueeze(0).long()

        return data


class ConvertRegionsToMultiClassd(MapTransform):
    """Invert the 3-channel overlapping-region encoding [ET, TC, WT] back into the
    single-channel mutually-exclusive class map (0=bg, 1=NCR, 2=ED, 3=ET).

    Used for pseudo-labels that were saved on disk in the old ET/TC/WT format, so
    they line up with the softmax fine-tuning targets produced by
    ``ConvertToMultiClassd``. The regions are nested (ET < TC < WT), so:
        WT-only            -> ED  (2)
        TC-but-not-ET      -> NCR (1)
        ET                 -> ET  (3)
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            regions = data[key]
            et = regions[0] > 0.5
            tc = regions[1] > 0.5
            wt = regions[2] > 0.5

            seg = torch.zeros_like(regions[0], dtype=torch.long)
            seg[wt] = 2   # edema (WT shell)
            seg[tc] = 1   # necrotic core (TC minus ET, refined next)
            seg[et] = 3   # enhancing tumour

            data[key] = seg.unsqueeze(0)

        return data

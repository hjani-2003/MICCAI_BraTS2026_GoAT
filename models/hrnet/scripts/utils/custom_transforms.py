from monai.transforms import MapTransform


class PrepareLabeld(MapTransform):
    """Normalizes the raw BraTS-GoAT segmentation label to a channel-first integer class map.

    Class indices (mutually exclusive, one per voxel):
        0 - bg  : background
        1 - ncr : necrotic / non-enhancing tumor core
        2 - ed  : peritumoral edema
        3 - et  : enhancing tumor

    Raw label values on disk already equal these class indices, so this is an
    identity mapping — it only normalizes shape/dtype for downstream transforms
    and the softmax/CE loss.
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            seg = data[key]
            if seg.shape[0] == 1:
                seg = seg[0]
            data[key] = seg.unsqueeze(0).float()

        return data

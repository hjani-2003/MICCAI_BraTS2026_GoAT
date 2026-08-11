

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.networks.utils import one_hot


class IgnoreAwareDiceCELoss(torch.nn.Module):



    def __init__(
        self,
        include_background=True,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        squared_pred=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
        ignore_index=4,
    ):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)


        self.dice = DiceLoss(
            include_background=bool(include_background),
            softmax=False,
            to_onehot_y=False,
            smooth_nr=float(smooth_nr),
            smooth_dr=float(smooth_dr),
            squared_pred=bool(squared_pred),
        )

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        target = target.long()
        if target.shape[1] != 1:
            raise ValueError(f"target must have a single channel, got shape {tuple(target.shape)}")

        valid = target != self.ignore_index  # (B, 1, X, Y, Z)


        ce_target = target[:, 0]  # (B, X, Y, Z)
        if valid.any():
            ce = F.cross_entropy(logits, ce_target, ignore_index=self.ignore_index)
        else:
            
            ce = logits.sum() * 0.0


        probs = torch.softmax(logits, dim=1)
        clamped = target.clone()
        clamped[~valid] = 0  # park ignored voxels on background so one_hot is in range
        onehot = one_hot(clamped, num_classes=num_classes)  # (B, C, X, Y, Z)
        mask = valid.to(probs.dtype)  # (B, 1, X, Y, Z), broadcasts over channels
        dice = self.dice(probs * mask, onehot * mask)

        return self.lambda_dice * dice + self.lambda_ce * ce


class RegionDiceBCELoss(torch.nn.Module):
    """Sum of BCE and soft Dice over the overlapping regions [ET, TC, WT], averaged.

    Targets are the 3-channel float masks produced by ConvertToBraTSRegionsd. The regions
    overlap (ET < TC < WT), so each channel is an independent sigmoid rather than one
    softmax over mutually-exclusive classes.
    """

    def __init__(
        self,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        squared_pred=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    ):
        super().__init__()
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)
        self.dice = DiceLoss(
            include_background=True,
            sigmoid=True,
            to_onehot_y=False,
            smooth_nr=float(smooth_nr),
            smooth_dr=float(smooth_dr),
            squared_pred=bool(squared_pred),
        )
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        if target.shape[1] != logits.shape[1]:
            raise ValueError(
                f"region target must have {logits.shape[1]} channels [ET, TC, WT], "
                f"got shape {tuple(target.shape)}. Is the dataloader using "
                f"ConvertToBraTSRegionsd?"
            )
        target = target.to(logits.dtype)
        return self.lambda_dice * self.dice(logits, target) + self.lambda_ce * self.bce(logits, target)

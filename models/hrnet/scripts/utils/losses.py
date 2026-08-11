import torch
import torch.nn as nn
from monai.losses import DiceLoss


class DiceCELoss(nn.Module):
    """Weighted combination of CrossEntropy and Dice losses using softmax activations.

    Both losses operate on raw logits (no prior softmax needed on input).
    Targets are integer class-index maps of shape [B, 1, D, H, W]; the Dice
    term one-hots them internally via `to_onehot_y`.
    Default weighting: 0.7 * CE + 0.3 * Dice.
    """

    def __init__(
        self,
        ce_weight: float = 0.7,
        dice_weight: float = 0.3,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        squared_pred: bool = True,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(
            softmax=True,
            to_onehot_y=True,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            squared_pred=squared_pred,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        ce_loss = self.ce(logits, targets.squeeze(1))
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

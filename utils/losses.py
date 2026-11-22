import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Dice Loss"""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds and targets expected in [B, 1, H, W] with values in [0, 1]
        preds = torch.clamp(preds, min=1e-7, max=1.0 - 1e-7)
        targets = targets.float()

        bce_loss = self.bce(preds, targets)

        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return bce_loss + dice_loss
import torch


def compute_segmentation_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> dict[str, float]:
    """Compute mIoU, DSC, Accuracy, Specificity, and Sensitivity on binary masks."""
    pred_bin = (preds >= threshold).float().view(-1)
    target_bin = (targets >= threshold).float().view(-1)

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1.0 - target_bin)).sum().item()
    fn = ((1.0 - pred_bin) * target_bin).sum().item()
    tn = ((1.0 - pred_bin) * (1.0 - target_bin)).sum().item()

    iou = (tp + eps) / (tp + fp + fn + eps)
    dsc = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    spec = (tn + eps) / (tn + fp + eps)
    sens = (tp + eps) / (tp + fn + eps)

    return {
        "mIoU": iou * 100.0,
        "DSC": dsc * 100.0,
        "Accuracy": acc * 100.0,
        "Specificity": spec * 100.0,
        "Sensitivity": sens * 100.0,
    }
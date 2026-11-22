import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.aulunet import aulunet_v1, aulunet_v2, aulunet_v3
from utils.dataset import ISICSegmentationDataset


def get_model(variant: str) -> torch.nn.Module:
    variants = {
        "v1": aulunet_v1,
        "v2": aulunet_v2,
        "v3": aulunet_v3,
    }
    if variant not in variants:
        raise ValueError(f"Unsupported variant '{variant}'. Choose from {list(variants.keys())}.")
    return variants[variant]()


def compute_metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    eps = 1e-7
    dsc = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    prec = tp / (tp + fp + eps)
    return {
        "DSC": dsc * 100.0,
        "mIoU": iou * 100.0,
        "Accuracy": acc * 100.0,
        "Sensitivity": sens * 100.0,
        "Specificity": spec * 100.0,
        "Precision": prec * 100.0,
    }


def save_overlay(img_chw: torch.Tensor, gt_hw: np.ndarray, pred_hw: np.ndarray, out_path: Path) -> None:
    img_rgb = (img_chw.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    overlay = img_bgr.copy()
    # Green = Ground Truth, Red = Prediction
    overlay[gt_hw == 1, 1] = np.minimum(255, overlay[gt_hw == 1, 1].astype(np.int32) + 90).astype(np.uint8)
    overlay[pred_hw == 1, 2] = np.minimum(255, overlay[pred_hw == 1, 2].astype(np.int32) + 90).astype(np.uint8)

    gt_vis = (gt_hw * 255).astype(np.uint8)
    pred_vis = (pred_hw * 255).astype(np.uint8)
    gt_vis_bgr = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
    pred_vis_bgr = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

    panel = np.hstack([img_bgr, gt_vis_bgr, pred_vis_bgr, overlay])
    cv2.imwrite(str(out_path), panel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained AULUNet checkpoint on ISIC test set.")
    parser.add_argument("--test-images", type=Path, default=Path("data/ISIC2018_256/test/images"))
    parser.add_argument("--test-masks", type=Path, default=Path("data/ISIC2018_256/test/masks"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/aulunet_v3_best.pth"))
    parser.add_argument("--variant", type=str, default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-samples", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ISICSegmentationDataset(args.test_images, args.test_masks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(args.variant).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out_dir / "overlays"
    if args.save_samples > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    total_tp = total_fp = total_fn = total_tn = 0.0
    per_image_rows = []
    saved_count = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            probs = outputs.squeeze(1)
            preds = (probs >= args.threshold).float()
            targets = (masks.squeeze(1) >= 0.5).float()

            for i in range(images.size(0)):
                p = preds[i].cpu().numpy().astype(np.uint8)
                t = targets[i].cpu().numpy().astype(np.uint8)

                tp = float(np.sum((p == 1) & (t == 1)))
                fp = float(np.sum((p == 1) & (t == 0)))
                fn = float(np.sum((p == 0) & (t == 1)))
                tn = float(np.sum((p == 0) & (t == 0)))

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                sample_name = f"sample_{batch_idx:04d}_{i}"
                m = compute_metrics_from_counts(tp, fp, fn, tn)
                m["name"] = sample_name
                per_image_rows.append(m)

                if saved_count < args.save_samples:
                    save_overlay(images[i], t, p, vis_dir / f"{sample_name}_eval.png")
                    saved_count += 1

    global_metrics = compute_metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    metric_keys = ["DSC", "mIoU", "Accuracy", "Sensitivity", "Specificity", "Precision"]
    mean_metrics = {
        k: float(np.mean([row[k] for row in per_image_rows])) if per_image_rows else 0.0
        for k in metric_keys
    }

    print("\n=== Test Set Evaluation (Per-Image Mean / Macro) ===")
    for k in metric_keys:
        print(f"{k:12s}: {mean_metrics[k]:.2f}%")

    print("\n=== Test Set Evaluation (Dataset-Wide / Micro) ===")
    for k in metric_keys:
        print(f"{k:12s}: {global_metrics[k]:.2f}%")

    csv_path = args.out_dir / "test_metrics.csv"
    fieldnames = ["name"] + metric_keys
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"Saved per-image metrics to {csv_path}")


if __name__ == "__main__":
    main()
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.aulunet import (
    baseline,
    m1,
    m2,
    m3,
    m4,
    m5,
    m6,
    aulunet_v1,
    aulunet_v2,
    aulunet_v3,
)
from utils.dataset import ISICSegmentationDataset


VARIANT_SPECS = [
    ("baseline", "Baseline (No AKF/RGF/LSG)", baseline, "baseline_best.pth"),
    ("m1", "M1 (+AKF)", m1, "aulunet_m1_best.pth"),
    ("m2", "M2 (+RGF)", m2, "aulunet_m2_best.pth"),
    ("m3", "M3 (+LSG)", m3, "aulunet_m3_best.pth"),
    ("m4", "M4 (+AKF +RGF)", m4, "aulunet_m4_best.pth"),
    ("m5", "M5 (+AKF +LSG)", m5, "aulunet_m5_best.pth"),
    ("m6", "M6 (+RGF +LSG)", m6, "aulunet_m6_best.pth"),
    ("v1", "AULUNet-v1 [4,8,16,24,36,60]", aulunet_v1, "aulunet_v1_best.pth"),
    ("v2", "AULUNet-v2 [4,8,16,24,40,60]", aulunet_v2, "aulunet_v2_best.pth"),
    ("v3", "AULUNet-v3 [5,10,15,25,35,60]", aulunet_v3, "aulunet_v3_best.pth"),
]

METRIC_KEYS = ["DSC", "mIoU", "Accuracy", "Sensitivity", "Specificity", "Precision"]


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


def resolve_checkpoint(ckpt_dir: Path, variant_key: str, expected_name: str) -> Path | None:
    variant_folder = ckpt_dir / variant_key
    primary = variant_folder / expected_name
    if primary.is_file():
        return primary

    if variant_folder.is_dir():
        best_matches = sorted(variant_folder.glob("*best*.pth"))
        if best_matches:
            return best_matches[0]
        any_pth = sorted(variant_folder.glob("*.pth"))
        if any_pth:
            return any_pth[0]

    flat_fallback = ckpt_dir / expected_name
    if flat_fallback.is_file():
        return flat_fallback

    return None


@torch.no_grad()
def evaluate_single_variant(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float]], float]:
    model.eval()

    # Warmup pass for accurate GPU latency timing
    dummy = torch.randn(1, 3, 256, 256, device=device)
    for _ in range(5):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_tp = total_fp = total_fn = total_tn = 0.0
    per_image_rows: list[dict[str, float]] = []
    total_infer_time = 0.0
    total_images = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_infer_time += time.perf_counter() - t0

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        probs = outputs.squeeze(1)
        preds = (probs >= threshold).float()
        targets = (masks.squeeze(1) >= 0.5).float()

        batch_size = images.size(0)
        total_images += batch_size

        for i in range(batch_size):
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

            row = compute_metrics_from_counts(tp, fp, fn, tn)
            row["sample_id"] = f"sample_{batch_idx:04d}_{i}"
            per_image_rows.append(row)

    micro_metrics = compute_metrics_from_counts(total_tp, total_fp, total_fn, total_tn)
    macro_metrics = {
        k: float(np.mean([r[k] for r in per_image_rows])) if per_image_rows else 0.0
        for k in METRIC_KEYS
    }
    avg_ms_per_img = (total_infer_time / max(total_images, 1)) * 1000.0
    return macro_metrics, micro_metrics, per_image_rows, avg_ms_per_img


def print_summary_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 112}")
    print(f" {title}")
    print(f"{'=' * 112}")
    header = (
        f"{'Variant':<10} | {'Configuration':<29} | {'Params':>8} | {'BestEp':>6} | "
        f"{'DSC':>7} | {'mIoU':>7} | {'Acc':>7} | {'Sens':>7} | {'Spec':>7} | {'Prec':>7} | {'ms/img':>6}"
    )
    print(header)
    print("-" * 112)
    for r in rows:
        print(
            f"{r['variant']:<10} | {r['description']:<29} | {r['params']:>8,d} | {str(r['best_epoch']):>6} | "
            f"{r['DSC']:>6.2f}% | {r['mIoU']:>6.2f}% | {r['Accuracy']:>6.2f}% | "
            f"{r['Sensitivity']:>6.2f}% | {r['Specificity']:>6.2f}% | {r['Precision']:>6.2f}% | {r['ms_per_img']:>6.2f}"
        )
    print(f"{'=' * 112}")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "variant",
        "description",
        "params",
        "best_epoch",
        "val_best_dsc",
        "DSC",
        "mIoU",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "ms_per_img",
        "checkpoint",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ablation study evaluation on ISIC test set.")
    parser.add_argument("--test-images", type=Path, default=Path("data/ISIC2018_256/test/images"))
    parser.add_argument("--test-masks", type=Path, default=Path("data/ISIC2018_256/test/masks"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/ablation_250ep"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results/ablation_250ep"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-per-image", action="store_true", help="Also save per-image CSVs for each variant")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ISICSegmentationDataset(args.test_images, args.test_masks)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Loaded {len(dataset)} test images from {args.test_images}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_per_image:
        (args.out_dir / "per_image").mkdir(parents=True, exist_ok=True)

    macro_rows: list[dict] = []
    micro_rows: list[dict] = []

    for variant_key, desc, builder_fn, expected_ckpt in VARIANT_SPECS:
        ckpt_path = resolve_checkpoint(args.ckpt_dir, variant_key, expected_ckpt)
        if ckpt_path is None:
            print(f"[SKIP] {variant_key:<10} -> No checkpoint found in {args.ckpt_dir / variant_key}")
            continue

        model = builder_fn().to(device)
        num_params = sum(p.numel() for p in model.parameters())

        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            best_epoch = ckpt.get("epoch", "-")
            val_best_dsc = round(float(ckpt["best_dsc"]), 2) if "best_dsc" in ckpt else "-"
        else:
            model.load_state_dict(ckpt)
            best_epoch = "-"
            val_best_dsc = "-"

        print(f"[EVAL] {variant_key:<10} ({num_params:,} params) <- {ckpt_path}")
        macro_m, micro_m, per_image_rows, ms_per_img = evaluate_single_variant(
            model, loader, device, args.threshold
        )

        common_info = {
            "variant": variant_key,
            "description": desc,
            "params": num_params,
            "best_epoch": best_epoch,
            "val_best_dsc": val_best_dsc,
            "ms_per_img": round(ms_per_img, 3),
            "checkpoint": str(ckpt_path),
        }

        macro_rows.append({**common_info, **{k: round(macro_m[k], 4) for k in METRIC_KEYS}})
        micro_rows.append({**common_info, **{k: round(micro_m[k], 4) for k in METRIC_KEYS}})

        if args.save_per_image:
            per_img_csv = args.out_dir / "per_image" / f"{variant_key}_test_metrics.csv"
            with open(per_img_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_id"] + METRIC_KEYS)
                writer.writeheader()
                writer.writerows(per_image_rows)

    if not macro_rows:
        print("\nNo checkpoints were evaluated. Verify paths under --ckpt-dir.")
        return

    print_summary_table("ABLATION STUDY — TEST SET (PER-IMAGE MEAN / MACRO)", macro_rows)
    print_summary_table("ABLATION STUDY — TEST SET (DATASET-WIDE / MICRO)", micro_rows)

    macro_csv = args.out_dir / "ablation_summary_macro.csv"
    micro_csv = args.out_dir / "ablation_summary_micro.csv"
    write_summary_csv(macro_csv, macro_rows)
    write_summary_csv(micro_csv, micro_rows)
    print(f"\nSaved macro summary CSV -> {macro_csv}")
    print(f"Saved micro summary CSV -> {micro_csv}")


if __name__ == "__main__":
    main()
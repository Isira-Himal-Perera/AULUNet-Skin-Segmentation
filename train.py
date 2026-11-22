import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from models.aulunet import aulunet_v1, aulunet_v2, aulunet_v3
from utils.dataset import ISICSegmentationDataset
from utils.losses import BCEDiceLoss
from utils.metrics import compute_segmentation_metrics


def get_model(variant: str):
    variants = {
        "v1": aulunet_v1,
        "v2": aulunet_v2,
        "v3": aulunet_v3,
    }
    if variant not in variants:
        raise ValueError(f"Unsupported variant '{variant}'. Choose from {list(variants.keys())}.")
    return variants[variant]()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    metric_sums = {"mIoU": 0.0, "DSC": 0.0, "Accuracy": 0.0, "Specificity": 0.0, "Sensitivity": 0.0}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        preds = model(images)
        loss = criterion(preds, masks)
        running_loss += loss.item() * images.size(0)

        batch_metrics = compute_segmentation_metrics(preds, masks)
        for k in metric_sums:
            metric_sums[k] += batch_metrics[k] * images.size(0)

    n = len(loader.dataset)
    avg_loss = running_loss / n
    avg_metrics = {k: v / n for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train AULUNet for skin lesion segmentation")
    parser.add_argument("--train-images", type=Path, required=True, help="Path to training images")
    parser.add_argument("--train-masks", type=Path, required=True, help="Path to training masks")
    parser.add_argument("--val-images", type=Path, required=True, help="Path to validation images")
    parser.add_argument("--val-masks", type=Path, required=True, help="Path to validation masks")
    parser.add_argument("--variant", type=str, default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ISICSegmentationDataset(
        args.train_images, args.train_masks, img_size=args.img_size, is_train=True
    )
    val_dataset = ISICSegmentationDataset(
        args.val_images, args.val_masks, img_size=args.img_size, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = get_model(args.variant).to(device)
    criterion = BCEDiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_dsc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mIoU: {val_metrics['mIoU']:.2f}% | "
            f"DSC: {val_metrics['DSC']:.2f}%"
        )

        if val_metrics["DSC"] > best_dsc:
            best_dsc = val_metrics["DSC"]
            ckpt_path = args.save_dir / f"aulunet_{args.variant}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "variant": args.variant,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dsc": best_dsc,
                },
                ckpt_path,
            )

    print(f"Training complete. Best Validation DSC: {best_dsc:.2f}%")


if __name__ == "__main__":
    main()
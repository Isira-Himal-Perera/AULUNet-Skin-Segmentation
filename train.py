import argparse
from pathlib import Path
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
from utils.losses import BCEDiceLoss
from utils.metrics import compute_segmentation_metrics


VARIANT_BUILDERS = {
    "baseline": baseline,
    "m1": m1,
    "m2": m2,
    "m3": m3,
    "m4": m4,
    "m5": m5,
    "m6": m6,
    "v1": aulunet_v1,
    "v2": aulunet_v2,
    "v3": aulunet_v3,
}


def get_model(variant: str):
    key = variant.lower()
    if key not in VARIANT_BUILDERS:
        raise ValueError(
            f"Unsupported variant '{variant}'. Choose from {list(VARIANT_BUILDERS.keys())}."
        )
    return VARIANT_BUILDERS[key]()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    total_iou = 0.0
    total_dsc = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)

        metrics = compute_segmentation_metrics(outputs, masks)
        if isinstance(metrics, dict):
            iou = metrics["mIoU"]
            dsc = metrics["DSC"]
        else:
            iou, dsc = metrics[0], metrics[1]

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_iou += iou * batch_size
        total_dsc += dsc * batch_size
        total_samples += batch_size

    denom = max(total_samples, 1)
    return (
        running_loss / denom,
        total_iou / denom,
        total_dsc / denom,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train AULUNet variants on ISIC segmentation")
    parser.add_argument("--train-images", type=str, required=True, help="Path to training images directory")
    parser.add_argument("--train-masks", type=str, required=True, help="Path to training masks directory")
    parser.add_argument("--val-images", type=str, required=True, help="Path to validation images directory")
    parser.add_argument("--val-masks", type=str, required=True, help="Path to validation masks directory")
    parser.add_argument(
        "--variant",
        type=str.lower,
        default="v3",
        choices=list(VARIANT_BUILDERS.keys()),
        help="Model variant to train",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Input image resolution")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate for CosineAnnealingLR")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save best checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    train_dataset = ISICSegmentationDataset(
        images_dir=args.train_images,
        masks_dir=args.train_masks,
        img_size=args.img_size,
        is_train=True,
    )
    val_dataset = ISICSegmentationDataset(
        images_dir=args.val_images,
        masks_dir=args.val_masks,
        img_size=args.img_size,
        is_train=False,
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
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        f"{args.variant}_best.pth"
        if args.variant == "baseline"
        else f"aulunet_{args.variant}_best.pth"
    )
    best_ckpt_path = save_dir / ckpt_name

    best_dsc = 0.0

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_dsc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mIoU: {val_iou:.2f}% | "
            f"DSC: {val_dsc:.2f}%",
            flush=True,
        )

        if val_dsc > best_dsc:
            best_dsc = val_dsc
            torch.save(
                {
                    "epoch": epoch,
                    "variant": args.variant,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dsc": best_dsc,
                },
                best_ckpt_path,
            )

    print(f"Training complete. Best Validation DSC: {best_dsc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
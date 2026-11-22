from pathlib import Path
import cv2


def preprocess_split(src_img_dir: Path, src_mask_dir: Path, dst_img_dir: Path, dst_mask_dir: Path, size: int = 256):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(src_img_dir.glob("*.jpg")):
        mask_path = src_mask_dir / f"{img_path.stem}_segmentation.png"
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(dst_img_dir / f"{img_path.stem}.png"), img_resized)
        cv2.imwrite(str(dst_mask_dir / f"{img_path.stem}_segmentation.png"), mask_resized)


if __name__ == "__main__":
    base = Path("data/ISIC2018")
    out = Path("data/ISIC2018_256")
    for split in ["train", "val", "test"]:
        preprocess_split(base / split / "images", base / split / "masks", out / split / "images", out / split / "masks")
    print("Done pre-resizing to 256x256.")
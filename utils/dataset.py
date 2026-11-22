from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class ISICSegmentationDataset(Dataset):
    """Dataset loader for paired skin lesion images and binary masks."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        img_size: int = 256,
        is_train: bool = True,
        mask_suffix: str = "_segmentation",
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.mask_suffix = mask_suffix

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.image_paths = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in exts]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def _find_mask_path(self, img_path: Path) -> Path:
        # Supports exact stem match, ISIC '*_segmentation.png', or common extensions
        candidates = [
            self.masks_dir / f"{img_path.stem}{self.mask_suffix}.png",  # <masks_dir>/<stem><mask_suffix>.png
            self.masks_dir / f"{img_path.stem}.png",                    # <masks_dir>/<stem>.png
            self.masks_dir / f"{img_path.stem}.jpg",                    # <masks_dir>/<stem>.jpg
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Mask not found for image: {img_path.name}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self._find_mask_path(img_path)

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img_pil = Image.fromarray(img_rgb)
        mask_pil = Image.fromarray(mask)

        # Resize to 256x256
        img_pil = TF.resize(img_pil, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask_pil = TF.resize(mask_pil, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)

        # Preprocessing with flips and rotations for training
        if self.is_train:
            if random.random() < 0.5:
                img_pil = TF.hflip(img_pil)
                mask_pil = TF.hflip(mask_pil)
            if random.random() < 0.5:
                img_pil = TF.vflip(img_pil)
                mask_pil = TF.vflip(mask_pil)
            if random.random() < 0.5:
                angle = random.uniform(-30.0, 30.0)
                img_pil = TF.rotate(img_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Convert to tensor and normalize  
        img_tensor = TF.to_tensor(img_pil)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_np = np.array(mask_pil, dtype=np.float32)
        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)

        return img_tensor, mask_tensor
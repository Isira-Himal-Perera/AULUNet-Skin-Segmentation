import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.aulunet import aulunet_v1, aulunet_v2, aulunet_v3


def get_model(variant: str) -> torch.nn.Module:
    variants = {
        "v1": aulunet_v1,
        "v2": aulunet_v2,
        "v3": aulunet_v3,
    }
    if variant not in variants:
        raise ValueError(f"Unsupported variant '{variant}'. Choose from {list(variants.keys())}.")
    return variants[variant]()


def load_and_prepare_image(image_path: Path, size: int = 256) -> tuple[np.ndarray, torch.Tensor]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    bgr_resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb_resized = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0)
    return bgr_resized, tensor


def build_side_by_side(orig_bgr: np.ndarray, mask_hw: np.ndarray, include_overlay: bool = False) -> np.ndarray:
    mask_vis_bgr = cv2.cvtColor((mask_hw * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    panels = [orig_bgr, mask_vis_bgr]

    if include_overlay:
        overlay = orig_bgr.copy()
        # Highlight predicted lesion region in red
        overlay[mask_hw == 1, 2] = np.minimum(255, overlay[mask_hw == 1, 2].astype(np.int32) + 100).astype(np.uint8)
        panels.append(overlay)

    return np.hstack(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AULUNet inference on a custom image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/aulunet_v3_best.pth"))
    parser.add_argument("--variant", type=str, default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay", action="store_true", help="Include a third overlay panel")
    parser.add_argument("--output", type=Path, default=Path("custom_prediction.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_bgr, tensor = load_and_prepare_image(args.image, size=args.size)
    tensor = tensor.to(device)

    model = get_model(args.variant).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        probs = outputs.squeeze().detach().cpu().numpy()
        mask_hw = (probs >= args.threshold).astype(np.uint8)

    result_panel = build_side_by_side(orig_bgr, mask_hw, include_overlay=args.overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), result_panel)
    print(f"Saved side-by-side output to {args.output}")


if __name__ == "__main__":
    main()
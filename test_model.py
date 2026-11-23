import torch
from models.aulunet import (
    aulunet_v1,
    aulunet_v2,
    aulunet_v3,
)

from models.m5 import (
    m5
)

from models.m4 import (
    m4
)


def test_models():
    x = torch.randn(2, 3, 256, 256)  # Batch of 2, RGB, 256x256[cite: 6]

    models = [
        ("M4        ", m4),
        ("M5        ", m5),
        ("AULUNet-v1", aulunet_v1),
        ("AULUNet-v2", aulunet_v2),
        ("AULUNet-v3", aulunet_v3),
    ]

    for name, model_fn in models:
        model = model_fn()
        model.eval()

        with torch.no_grad():
            out = model(x)

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(
            f"[{name}] Output Shape: {out.shape} | Parameters: {params:.3f}M"
        )


if __name__ == "__main__":
    test_models()
import torch
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

def test_models():
    x = torch.randn(2, 3, 256, 256)  # Batch of 2, RGB, 256x256

    models = [
        ("Baseline", baseline),
        ("M1", m1),
        ("M2", m2),
        ("M3", m3),
        ("M4", m4),
        ("M5", m5),
        ("M6", m6),
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
            f"[{name:^10}] Output Shape: {out.shape} | Parameters: {params:.3f}M"
        )


if __name__ == "__main__":
    test_models()
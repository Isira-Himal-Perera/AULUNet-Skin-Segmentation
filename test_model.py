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

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def test_models():
    if not HAS_THOP:
        print("Warning: 'thop' is not installed. Install it via `pip install thop` to compute GFLOPs.\n")

    # Profile with batch_size=1 for standard per-image GFLOPs;
    # use `x = torch.randn(2, 3, 256, 256)` if you want batch=2 GFLOPs.
    x = torch.randn(1, 3, 256, 256)

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

        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        if HAS_THOP:
            # thop returns MACs; multiply by 2 for FLOPs (1 MAC = 2 FLOPs: 1 mul + 1 add)
            macs, _ = profile(model, inputs=(x,), verbose=False)
            gflops = (macs * 2) / 1e9
            print(
                f"[{name:^10}] Output Shape: {tuple(out.shape)} | "
                f"Params: {params_m:.3f}M | GFLOPs: {gflops:.3f}"
            )
        else:
            print(
                f"[{name:^10}] Output Shape: {tuple(out.shape)} | "
                f"Params: {params_m:.3f}M | GFLOPs: N/A"
            )


if __name__ == "__main__":
    test_models()
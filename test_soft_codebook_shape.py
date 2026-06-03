from types import SimpleNamespace

import torch

from models.soft_codebook import (
    SoftCodebookAdapterList,
    apply_soft_codebook_flat_if_enabled,
    apply_soft_codebook_if_enabled,
)


def main():
    args = SimpleNamespace(use_soft_codebook=True)
    channels = [8, 16, 32]
    features = [
        torch.randn(2, 8, 14, 14),
        torch.randn(2, 16, 7, 7),
        torch.randn(2, 32, 5, 5),
    ]
    soft_codebook = SoftCodebookAdapterList(
        channels,
        num_embeddings=32,
        tau=0.2,
        gamma=0.03,
        warmup_epochs=5,
        conf_gate=True,
        gate_threshold=0.0,
        gate_temp=0.05,
    )

    out0 = apply_soft_codebook_if_enabled(args, soft_codebook, features, epoch=0)
    out5 = apply_soft_codebook_if_enabled(args, soft_codebook, features, epoch=5)
    for before, after0, after5 in zip(features, out0, out5):
        assert before.shape == after0.shape == after5.shape
        assert torch.allclose(before, after0)
        assert torch.isfinite(after0).all()
        assert torch.isfinite(after5).all()

    flat = features[0].permute(0, 2, 3, 1).reshape(-1, channels[0])
    flat_out0 = apply_soft_codebook_flat_if_enabled(args, soft_codebook, 0, flat, epoch=0)
    soft_codebook.zero_grad()
    flat_out0.pow(2).mean().backward()
    assert soft_codebook.adapters[0].codebook.weight.grad is not None
    assert soft_codebook.adapters[0].codebook.weight.grad.abs().sum().item() == 0.0

    flat_out = apply_soft_codebook_flat_if_enabled(args, soft_codebook, 0, flat, epoch=5)
    assert flat.shape == flat_out.shape
    assert torch.isfinite(flat_out).all()
    soft_codebook.zero_grad()
    flat_out.pow(2).mean().backward()
    assert soft_codebook.adapters[0].codebook.weight.grad is not None

    reloaded = SoftCodebookAdapterList(
        channels,
        num_embeddings=32,
        tau=0.2,
        gamma=0.03,
        warmup_epochs=5,
        conf_gate=True,
        gate_threshold=0.0,
        gate_temp=0.05,
    )
    reloaded.load_state_dict(soft_codebook.state_dict(), strict=False)


if __name__ == "__main__":
    main()
    print("Soft codebook shape test passed.")

from types import SimpleNamespace

import torch

from models.vq import MultiScaleVQ
from raw_vqops import apply_raw_vqops_if_enabled, train_raw_vqops_if_enabled


def test_raw_vqops_shape():
    args = SimpleNamespace(use_raw_vqops=True, raw_vq_debug=False)
    features = [
        torch.randn(2, 4, 4, 4),
        torch.randn(2, 5, 2, 2),
        torch.randn(2, 6, 1, 1),
    ]
    masks = [
        torch.zeros(2, 4, 4),
        torch.zeros(2, 2, 2),
        torch.zeros(2, 1, 1),
    ]
    raw_vq_ops = MultiScaleVQ(num_embeddings=8, channels=(4, 5, 6))
    optimizer_vq = torch.optim.Adam(raw_vq_ops.parameters(), lr=1e-4)

    loss_vq = train_raw_vqops_if_enabled(args, raw_vq_ops, optimizer_vq, features, masks)
    outputs = apply_raw_vqops_if_enabled(args, raw_vq_ops, features, prefix="shape_test", loss_vq=loss_vq)

    assert loss_vq is not None and torch.isfinite(loss_vq).item()
    assert [tuple(output.shape) for output in outputs] == [tuple(feature.shape) for feature in features]


if __name__ == "__main__":
    test_raw_vqops_shape()
    print("Raw VQOps shape test passed.")

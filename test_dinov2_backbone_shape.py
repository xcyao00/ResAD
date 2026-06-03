from unittest.mock import patch

import torch
import torch.nn as nn

from models.dinov2_backbone import DINOv2BackboneWrapper


class _FakeDINOv2(nn.Module):
    embed_dim = 384
    patch_size = 14

    def forward_features(self, images):
        batch_size = images.shape[0]
        patch_tokens = torch.randn(batch_size, 16 * 16, self.embed_dim, device=images.device)
        return {"x_norm_patchtokens": patch_tokens}

    def get_intermediate_layers(self, images, n, reshape=False, return_class_token=False, norm=True):
        batch_size = images.shape[0]
        return tuple(
            torch.randn(batch_size, 16 * 16, self.embed_dim, device=images.device)
            for _ in n
        )


def test_dinov2_backbone_shape():
    with patch("torch.hub.load", return_value=_FakeDINOv2()):
        encoder = DINOv2BackboneWrapper(model_name="dinov2_vits14").eval()
    with torch.no_grad():
        features = encoder(torch.randn(2, 3, 224, 224))
    expected_shapes = [(2, 40, 56, 56), (2, 72, 28, 28), (2, 200, 14, 14)]
    assert [tuple(feature.shape) for feature in features] == expected_shapes
    assert encoder.feature_info.channels() == [40, 72, 200]


def test_dinov2_intermediate_fixed_projected_shape():
    with patch("torch.hub.load", return_value=_FakeDINOv2()):
        encoder = DINOv2BackboneWrapper(
            model_name="dinov2_vits14",
            feature_mode="intermediate_fixed_projected",
            layers=(4, 8, 12),
            proj_dim=256,
        ).eval()
    with torch.no_grad():
        features = encoder(torch.randn(2, 3, 224, 224))
    expected_shapes = [(2, 256, 16, 16), (2, 256, 16, 16), (2, 256, 16, 16)]
    assert [tuple(feature.shape) for feature in features] == expected_shapes
    assert encoder.feature_info.channels() == [256, 256, 256]
    assert all(not param.requires_grad for param in encoder.projections.parameters())


def test_dinov2_intermediate_without_projection_shape():
    with patch("torch.hub.load", return_value=_FakeDINOv2()):
        encoder = DINOv2BackboneWrapper(
            model_name="dinov2_vits14",
            feature_mode="intermediate_fixed_projected",
            layers=(4, 8, 12),
            proj_dim=0,
        ).eval()
    with torch.no_grad():
        features = encoder(torch.randn(2, 3, 224, 224))
    expected_shapes = [(2, 384, 16, 16), (2, 384, 16, 16), (2, 384, 16, 16)]
    assert [tuple(feature.shape) for feature in features] == expected_shapes
    assert encoder.feature_info.channels() == [384, 384, 384]


if __name__ == "__main__":
    test_dinov2_backbone_shape()
    test_dinov2_intermediate_fixed_projected_shape()
    test_dinov2_intermediate_without_projection_shape()
    print("DINOv2 backbone shape test passed.")

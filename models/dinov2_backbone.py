import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DINOV2_BACKBONES = ("dinov2_vits14", "dinov2_vitb14")
DINOV2_FEATURE_MODES = ("final_projected", "intermediate_fixed_projected", "final_only")
_DINOV2_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return self._channels


class DINOv2BackboneWrapper(nn.Module):
    def __init__(
        self,
        model_name="dinov2_vits14",
        out_dims=(40, 72, 200),
        out_sizes=(56, 28, 14),
        freeze=True,
        feature_mode="final_projected",
        layers=(4, 8, 12),
        proj_dim=256,
    ):
        super().__init__()
        if model_name not in DINOV2_BACKBONES:
            raise ValueError(f"Unsupported DINOv2 model_name: {model_name}")
        if feature_mode not in DINOV2_FEATURE_MODES:
            raise ValueError(f"Unsupported DINOv2 feature_mode: {feature_mode}")

        self.model_name = model_name
        self.feature_mode = feature_mode
        self.out_dims = tuple(out_dims)
        self.out_sizes = tuple(out_sizes)
        self.freeze = freeze
        self.layers = tuple(int(layer) for layer in layers)
        self.layer_indices = tuple(layer - 1 for layer in self.layers)
        self.proj_dim = int(proj_dim)
        if self.proj_dim < 0:
            raise ValueError(f"dinov2_proj_dim must be >= 0, got {self.proj_dim}.")
        self.dino = torch.hub.load("facebookresearch/dinov2", model_name)
        self.dino.eval()

        if self.freeze:
            for param in self.dino.parameters():
                param.requires_grad = False

        self.embed_dim = self._infer_embed_dim()
        self.patch_size = self._infer_patch_size()
        if self.feature_mode == "final_projected":
            if len(self.out_dims) != 3 or len(self.out_sizes) != 3:
                raise ValueError("final_projected expects exactly 3 output levels.")
            self.projections = nn.ModuleList([
                nn.Conv2d(self.embed_dim, self.out_dims[0], kernel_size=1),
                nn.Conv2d(self.embed_dim, self.out_dims[1], kernel_size=1),
                nn.Conv2d(self.embed_dim, self.out_dims[2], kernel_size=1),
            ])
            self._init_projections(self.projections, trainable=False)
            feature_dims = self.out_dims
        elif self.feature_mode == "final_only":
            self.projections = nn.ModuleList()
            feature_dims = [self.embed_dim]
        else:
            if len(self.layers) != 3:
                raise ValueError("intermediate_fixed_projected expects exactly 3 DINOv2 layers.")
            if any(layer <= 0 for layer in self.layers):
                raise ValueError(f"DINOv2 layers are 1-indexed and must be positive, got {self.layers}.")
            if self.proj_dim > 0:
                self.projections = nn.ModuleList([
                    nn.Conv2d(self.embed_dim, self.proj_dim, kernel_size=1)
                    for _ in self.layers
                ])
                self._init_projections(self.projections, trainable=False)
                feature_dims = [self.proj_dim] * len(self.layers)
            else:
                self.projections = nn.ModuleList()
                feature_dims = [self.embed_dim] * len(self.layers)

        self.feature_info = _FeatureInfo(feature_dims)

    def _infer_embed_dim(self):
        for attr in ("embed_dim", "num_features"):
            value = getattr(self.dino, attr, None)
            if isinstance(value, int):
                return value
        if self.model_name in _DINOV2_EMBED_DIMS:
            return _DINOV2_EMBED_DIMS[self.model_name]
        raise ValueError(f"Could not infer DINOv2 embed dim for {self.model_name}")

    def _infer_patch_size(self):
        patch_size = getattr(self.dino, "patch_size", None)
        if isinstance(patch_size, int):
            return patch_size
        patch_embed = getattr(self.dino, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size, int):
            return patch_size
        if isinstance(patch_size, tuple):
            return patch_size[0]
        return 14

    def _init_projections(self, projections, trainable):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for projection in projections:
                nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)
                for param in projection.parameters():
                    param.requires_grad = bool(trainable)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def _forward_dino(self, images):
        if self.freeze:
            with torch.no_grad():
                return self.dino.forward_features(images)
        return self.dino.forward_features(images)

    def _get_intermediate_layers(self, images):
        kwargs = dict(
            n=list(self.layer_indices),
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        if self.freeze:
            with torch.no_grad():
                try:
                    return self.dino.get_intermediate_layers(images, **kwargs)
                except TypeError:
                    kwargs.pop("norm", None)
                    return self.dino.get_intermediate_layers(images, **kwargs)
        try:
            return self.dino.get_intermediate_layers(images, **kwargs)
        except TypeError:
            kwargs.pop("norm", None)
            return self.dino.get_intermediate_layers(images, **kwargs)

    def _extract_patch_tokens(self, outputs):
        if isinstance(outputs, dict):
            if "x_norm_patchtokens" in outputs:
                patch_tokens = outputs["x_norm_patchtokens"]
            elif "x_prenorm" in outputs:
                patch_tokens = outputs["x_prenorm"]
            else:
                raise ValueError("DINOv2 forward_features output has no patch-token tensor.")
        elif isinstance(outputs, (list, tuple)):
            if len(outputs) == 2 and torch.is_tensor(outputs[0]):
                patch_tokens = outputs[0]
            else:
                raise ValueError("Expected one DINOv2 feature tensor, got a sequence.")
        else:
            patch_tokens = outputs

        if patch_tokens.dim() != 3:
            raise ValueError(f"Expected patch tokens with shape [B, N, C], got {tuple(patch_tokens.shape)}")

        num_tokens = patch_tokens.shape[1]
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            cls_removed_tokens = num_tokens - 1
            cls_removed_grid = int(math.sqrt(cls_removed_tokens))
            if cls_removed_grid * cls_removed_grid != cls_removed_tokens:
                raise ValueError(f"DINOv2 patch token count must be square, got N={num_tokens}")
            patch_tokens = patch_tokens[:, 1:, :]

        return patch_tokens

    def _tokens_to_map(self, patch_tokens):
        batch_size, num_tokens, channels = patch_tokens.shape
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"DINOv2 patch token count must be square, got N={num_tokens}")
        return patch_tokens.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)

    def _forward_final_projected(self, images):
        outputs = self._forward_dino(images)
        patch_map = self._tokens_to_map(self._extract_patch_tokens(outputs))
        feat0 = F.interpolate(
            self.projections[0](patch_map),
            size=(self.out_sizes[0], self.out_sizes[0]),
            mode="bilinear",
            align_corners=False,
        )
        feat1 = F.interpolate(
            self.projections[1](patch_map),
            size=(self.out_sizes[1], self.out_sizes[1]),
            mode="bilinear",
            align_corners=False,
        )
        feat2 = F.interpolate(
            self.projections[2](patch_map),
            size=(self.out_sizes[2], self.out_sizes[2]),
            mode="bilinear",
            align_corners=False,
        )
        return [feat0, feat1, feat2]

    def _forward_final_only(self, images):
        outputs = self._forward_dino(images)
        patch_map = self._tokens_to_map(self._extract_patch_tokens(outputs))
        return [patch_map]

    def _forward_intermediate_fixed_projected(self, images):
        layer_outputs = self._get_intermediate_layers(images)
        if len(layer_outputs) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} DINOv2 layers, got {len(layer_outputs)}.")

        features = []
        for idx, layer_output in enumerate(layer_outputs):
            patch_map = self._tokens_to_map(self._extract_patch_tokens(layer_output))
            if self.proj_dim > 0:
                patch_map = self.projections[idx](patch_map)
            features.append(patch_map)
        return features

    def forward(self, images):
        if self.feature_mode == "final_projected":
            return self._forward_final_projected(images)
        if self.feature_mode == "final_only":
            return self._forward_final_only(images)
        if self.feature_mode == "intermediate_fixed_projected":
            return self._forward_intermediate_fixed_projected(images)
        raise ValueError(f"Unsupported DINOv2 feature_mode: {self.feature_mode}")

    @property
    def uses_resize(self):
        return self.feature_mode == "final_projected"

    def expected_feature_size(self, image_size=224):
        if self.feature_mode == "final_projected":
            return self.out_sizes
        grid_size = image_size // self.patch_size
        if self.feature_mode == "final_only":
            return (grid_size,)
        return (grid_size,) * len(self.layers)


def print_dinov2_config(encoder, image_size=224):
    print("[DINOv2] feature_mode:", encoder.feature_mode)
    if encoder.feature_mode == "final_only":
        print("[DINOv2] layers: final")
    else:
        print("[DINOv2] layers:", list(encoder.layers))
    print("[DINOv2] proj_dim:", encoder.proj_dim)
    print("[DINOv2] resize:", encoder.uses_resize)
    print("[DINOv2] feature dims:", encoder.feature_info.channels())
    sizes = encoder.expected_feature_size(image_size=image_size)
    if len(set(sizes)) == 1:
        print(f"[DINOv2] feature size: {sizes[0]}x{sizes[0]}")
    else:
        print("[DINOv2] feature size:", [f"{size}x{size}" for size in sizes])


def dinov2_shape_test(
    model_name="dinov2_vits14",
    device="cpu",
    feature_mode="final_projected",
    layers=(4, 8, 12),
    proj_dim=256,
):
    encoder = DINOv2BackboneWrapper(
        model_name=model_name,
        feature_mode=feature_mode,
        layers=layers,
        proj_dim=proj_dim,
    ).to(device).eval()
    images = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        features = encoder(images)

    if feature_mode == "final_projected":
        expected_shapes = [(2, 40, 56, 56), (2, 72, 28, 28), (2, 200, 14, 14)]
    elif feature_mode == "final_only":
        expected_shapes = [(2, encoder.embed_dim, 16, 16)]
    else:
        channels = encoder.embed_dim if proj_dim == 0 else proj_dim
        expected_shapes = [(2, channels, 16, 16)] * 3

    for feature, expected_shape in zip(features, expected_shapes):
        if tuple(feature.shape) != expected_shape:
            raise AssertionError(f"Expected {expected_shape}, got {tuple(feature.shape)}")
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="dinov2_vits14", choices=DINOV2_BACKBONES)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--feature_mode", type=str, default="final_projected", choices=DINOV2_FEATURE_MODES)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--proj_dim", type=int, default=256)
    args = parser.parse_args()
    dinov2_shape_test(
        model_name=args.model_name,
        device=args.device,
        feature_mode=args.feature_mode,
        layers=args.layers,
        proj_dim=args.proj_dim,
    )
    print("DINOv2 shape test passed.")

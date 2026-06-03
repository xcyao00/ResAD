import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return self._channels


_OPENAI_CLIP_URLS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

_OPENAI_NAME_ALIASES = {
    "ViT-B-32": "ViT-B/32",
    "ViT-B/16": "ViT-B/16",
    "ViT-B-16": "ViT-B/16",
    "ViT-L/14": "ViT-L/14",
    "ViT-L-14": "ViT-L/14",
    "ViT-L/14@336px": "ViT-L/14@336px",
    "ViT-L-14-336": "ViT-L/14@336px",
    "ViT-L-14-336px": "ViT-L/14@336px",
}


class CLIPRawFeatureExtractor(nn.Module):
    """
    Frozen CLIP ViT raw patch-token extractor.

    clip_layers are user-facing 1-indexed transformer block numbers. For example,
    --clip_layers 6 12 24 captures resblocks[5], resblocks[11], and resblocks[23].
    """

    def __init__(
        self,
        model_name="ViT-L-14-336",
        pretrained="openai",
        layers=(6, 12, 24),
        image_size=518,
        freeze=True,
        weight_source="open_clip",
        checkpoint="",
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for --feature_backbone clip_raw. "
                "Install it with `pip install open_clip_torch`."
            ) from exc

        self.model_name = model_name
        self.pretrained = pretrained
        if pretrained == "openai_local":
            weight_source = "openai_local"
        self.weight_source = weight_source
        self.checkpoint = checkpoint
        self.layers = tuple(int(layer) for layer in layers)
        self.layer_indices = tuple(layer - 1 for layer in self.layers)
        self.image_size = int(image_size)
        self.freeze = freeze

        self.model = self._build_model(open_clip)
        self.model.eval()
        self.visual = self.model.visual
        self.resblocks = self._get_resblocks()
        self._validate_layers()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.embed_dim = self._infer_embed_dim()
        self.patch_size = self._infer_patch_size()
        self.feature_info = _FeatureInfo([self.embed_dim] * len(self.layers))
        self._debug_printed = False

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1), persistent=False)

    def _build_model(self, open_clip):
        if self.weight_source == "open_clip":
            return self._build_open_clip_model(open_clip, pretrained=self.pretrained)
        if self.weight_source == "openai_local":
            model = self._build_open_clip_model(open_clip, pretrained=None)
            checkpoint_path = self._resolve_openai_checkpoint()
            print(f"Use pretrained model from openai: {checkpoint_path}")
            self._load_openai_checkpoint(model, checkpoint_path)
            return model
        raise ValueError(f"Unsupported clip_weight_source: {self.weight_source}")

    def _build_open_clip_model(self, open_clip, pretrained):
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=pretrained,
                force_image_size=self.image_size,
            )
        except TypeError:
            model, _, _ = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=pretrained,
            )
        return model

    def _resolve_openai_checkpoint(self):
        if self.checkpoint:
            checkpoint_path = os.path.expanduser(self.checkpoint)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"clip_checkpoint does not exist: {checkpoint_path}")
            return checkpoint_path

        openai_name = _OPENAI_NAME_ALIASES.get(self.model_name, self.model_name)
        if openai_name not in _OPENAI_CLIP_URLS:
            raise ValueError(
                f"No OpenAI CLIP URL registered for {self.model_name}. "
                "Use --clip_checkpoint to provide a local .pt file."
            )

        cache_dir = os.path.expanduser(os.environ.get("CLIP_CACHE_DIR", "~/.cache/clip"))
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(_OPENAI_CLIP_URLS[openai_name])
        checkpoint_path = os.path.join(cache_dir, filename)
        if not os.path.isfile(checkpoint_path):
            print(f"Downloading OpenAI CLIP weights to {checkpoint_path}")
            torch.hub.download_url_to_file(_OPENAI_CLIP_URLS[openai_name], checkpoint_path, progress=True)
        else:
            print(f"Using cached OpenAI CLIP weights: {checkpoint_path}")
        return checkpoint_path

    def _load_openai_checkpoint(self, model, checkpoint_path):
        state_dict = self._read_openai_state_dict(checkpoint_path)
        state_dict = self._strip_state_prefixes(state_dict)
        state_dict = self._resize_visual_positional_embedding(model, state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False)
        visual_missing = [key for key in incompatible.missing_keys if key.startswith("visual.")]
        if visual_missing:
            raise RuntimeError(
                "OpenAI CLIP checkpoint did not load all visual encoder weights. "
                f"Missing visual keys include: {visual_missing[:8]}"
            )
        missing = [key for key in incompatible.missing_keys if not key.startswith("visual.")]
        if missing:
            print(f"[CLIPRawFeatureExtractor] non-visual missing keys: {missing[:8]}")
        if incompatible.unexpected_keys:
            print(f"[CLIPRawFeatureExtractor] unexpected keys: {incompatible.unexpected_keys[:8]}")

    def _read_openai_state_dict(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    return checkpoint["state_dict"]
                return checkpoint
        except RuntimeError:
            checkpoint = torch.jit.load(checkpoint_path, map_location="cpu")
            return checkpoint.state_dict()
        if hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
        raise ValueError(f"Could not read OpenAI CLIP checkpoint: {checkpoint_path}")

    def _strip_state_prefixes(self, state_dict):
        stripped = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            stripped[key] = value
        return stripped

    def _resize_visual_positional_embedding(self, model, state_dict):
        key = "visual.positional_embedding"
        if key not in state_dict or not hasattr(model.visual, "positional_embedding"):
            return state_dict
        source = state_dict[key]
        target = model.visual.positional_embedding
        if tuple(source.shape) == tuple(target.shape):
            return state_dict
        if source.dim() != 2 or target.dim() != 2:
            return state_dict

        cls_pos = source[:1]
        patch_pos = source[1:]
        old_grid = int(math.sqrt(patch_pos.shape[0]))
        new_grid = int(math.sqrt(target.shape[0] - 1))
        if old_grid * old_grid != patch_pos.shape[0] or new_grid * new_grid != target.shape[0] - 1:
            return state_dict
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, source.shape[-1]).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos.float(), size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid * new_grid, source.shape[-1]).to(source.dtype)
        state_dict[key] = torch.cat([cls_pos, patch_pos], dim=0)
        return state_dict

    def _get_resblocks(self):
        transformer = getattr(self.visual, "transformer", None)
        resblocks = getattr(transformer, "resblocks", None)
        if resblocks is None:
            raise ValueError("CLIP visual transformer has no resblocks; only ViT CLIP models are supported.")
        return resblocks

    def _validate_layers(self):
        if not self.layers:
            raise ValueError("clip_layers must contain at least one layer.")
        num_blocks = len(self.resblocks)
        for layer in self.layers:
            if layer < 1 or layer > num_blocks:
                raise ValueError(f"clip layer {layer} is out of range for {num_blocks} transformer blocks.")

    def _infer_embed_dim(self):
        conv1 = getattr(self.visual, "conv1", None)
        out_channels = getattr(conv1, "out_channels", None)
        if isinstance(out_channels, int):
            return out_channels
        width = getattr(getattr(self.visual, "transformer", None), "width", None)
        if isinstance(width, int):
            return width
        raise ValueError("Could not infer CLIP visual embed dimension.")

    def _infer_patch_size(self):
        conv1 = getattr(self.visual, "conv1", None)
        kernel_size = getattr(conv1, "kernel_size", None)
        if isinstance(kernel_size, tuple):
            return kernel_size[0]
        if isinstance(kernel_size, int):
            return kernel_size
        return 14

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _normalize_for_clip(self, images):
        images = images * self.imagenet_std + self.imagenet_mean
        return (images - self.clip_mean) / self.clip_std

    def _to_bnc(self, tokens, batch_size):
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.dim() != 3:
            raise ValueError(f"Expected CLIP block output [B,N,C] or [N,B,C], got {tuple(tokens.shape)}")
        if tokens.shape[0] == batch_size and tokens.shape[1] != batch_size:
            return tokens
        if tokens.shape[1] == batch_size:
            return tokens.permute(1, 0, 2).contiguous()
        raise ValueError(f"Cannot infer CLIP token layout from shape {tuple(tokens.shape)} and batch={batch_size}")

    def _tokens_to_map(self, tokens):
        num_tokens = tokens.shape[1]
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            patch_tokens = num_tokens - 1
            patch_grid = int(math.sqrt(patch_tokens))
            if patch_grid * patch_grid != patch_tokens:
                raise ValueError(f"CLIP patch token count must be square after class-token removal, got N={num_tokens}")
            tokens = tokens[:, 1:, :]
            grid_size = patch_grid
        b, n, c = tokens.shape
        if grid_size * grid_size != n:
            raise ValueError(f"CLIP patch token count must be square, got N={n}")
        return tokens.transpose(1, 2).reshape(b, c, grid_size, grid_size)

    def _debug_shapes(self, features):
        if self._debug_printed:
            return
        print("feature_backbone = clip_raw")
        print(f"clip_model = {self.model_name}")
        print(f"clip_layers = {list(self.layers)}")
        for layer, feature in zip(self.layers, features):
            print(f"layer{layer} shape = {tuple(feature.shape)}")
        self._debug_printed = True

    def forward(self, images):
        batch_size = images.shape[0]
        captured = {}
        hooks = []

        def make_hook(layer):
            def hook(_module, _inputs, output):
                captured[layer] = self._to_bnc(output, batch_size).detach()
            return hook

        for layer, layer_idx in zip(self.layers, self.layer_indices):
            hooks.append(self.resblocks[layer_idx].register_forward_hook(make_hook(layer)))

        try:
            images = self._normalize_for_clip(images)
            if self.freeze:
                with torch.no_grad():
                    _ = self.visual(images)
            else:
                _ = self.visual(images)
        finally:
            for hook in hooks:
                hook.remove()

        features = []
        for layer in self.layers:
            if layer not in captured:
                raise RuntimeError(f"CLIP layer {layer} was not captured.")
            features.append(self._tokens_to_map(captured[layer]))
        self._debug_shapes(features)
        return features

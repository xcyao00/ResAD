import importlib
import json
import os
import subprocess
import sys
from urllib.parse import urlparse

import torch
import torch.nn as nn


_OPENAI_CLIP_BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = list(channels)

    def channels(self):
        return self._channels


class AdaCLIPPromptedFeatureExtractor(nn.Module):
    """
    Frozen AdaCLIP prompted patch-token extractor.

    clip_layers are user-facing 1-indexed transformer block numbers. The
    AdaCLIP_res model must expose:
        extract_prompted_features(images, layers, return_projected=False)
    and return a dict keyed by layer names such as "layer6".
    """

    def __init__(
        self,
        adaclip_repo_url="https://github.com/tomo082/AdaCLIP_res",
        adaclip_repo_path="",
        checkpoint="",
        checkpoint_url="",
        cache_dir="~/.cache/adaclip_res",
        model_name="ViT-L-14-336",
        layers=(6, 12, 24),
        image_size=336,
        return_projected=False,
        freeze=True,
        device="cuda:0",
    ):
        super().__init__()
        self.adaclip_repo_url = adaclip_repo_url
        self.adaclip_repo_path = adaclip_repo_path
        self.checkpoint = checkpoint
        self.checkpoint_url = checkpoint_url
        self.cache_dir = os.path.expanduser(cache_dir)
        self.model_name = model_name
        self.layers = tuple(int(layer) for layer in layers)
        self.image_size = int(image_size)
        self.return_projected = bool(return_projected)
        self.freeze = freeze
        self.device_name = device
        self._debug_printed = False

        if len(self.layers) != 3:
            raise ValueError("adaclip_prompted currently expects exactly 3 clip_layers for ResAD multi-level features.")
        if not self.checkpoint and not self.checkpoint_url:
            raise ValueError(
                "--feature_backbone adaclip_prompted requires --adaclip_checkpoint "
                "or --adaclip_checkpoint_url."
            )

        repo_path = self._resolve_repo_path()
        checkpoint_path = self._resolve_checkpoint_path()
        self.repo_path = repo_path
        self.checkpoint_path = checkpoint_path

        self.trainer = self._build_adaclip_trainer(repo_path, checkpoint_path)
        self.model = self.trainer.clip_model
        if not hasattr(self.model, "extract_prompted_features"):
            raise AttributeError(
                "AdaCLIP_res model does not expose extract_prompted_features(...). "
                "Update tomo082/AdaCLIP_res with the prompted feature extraction API first."
            )

        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.trainer.parameters():
                param.requires_grad = False

        channels = self._infer_channels(repo_path)
        self.feature_info = _FeatureInfo([channels] * len(self.layers))

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1), persistent=False)

    def _resolve_repo_path(self):
        if self.adaclip_repo_path:
            repo_path = os.path.abspath(os.path.expanduser(self.adaclip_repo_path))
            if not os.path.isdir(repo_path):
                raise FileNotFoundError(f"adaclip_repo_path does not exist: {repo_path}")
            return repo_path

        if not self.adaclip_repo_url:
            raise ValueError("--adaclip_repo_url is required when --adaclip_repo_path is empty.")

        repo_dir = os.path.join(self.cache_dir, "repos", "AdaCLIP_res")
        if os.path.isdir(repo_dir):
            return repo_dir

        os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", self.adaclip_repo_url, repo_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to clone AdaCLIP_res. Provide --adaclip_repo_path with a local clone, "
                f"or check --adaclip_repo_url. url={self.adaclip_repo_url}"
            ) from exc
        return repo_dir

    def _resolve_checkpoint_path(self):
        if self.checkpoint:
            checkpoint_path = os.path.abspath(os.path.expanduser(self.checkpoint))
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"adaclip_checkpoint does not exist: {checkpoint_path}")
            return checkpoint_path

        os.makedirs(self.cache_dir, exist_ok=True)
        parsed = urlparse(self.checkpoint_url)
        filename = os.path.basename(parsed.path)
        if not filename:
            raise ValueError(f"Could not infer checkpoint filename from URL: {self.checkpoint_url}")
        checkpoint_path = os.path.join(self.cache_dir, filename)
        if os.path.isfile(checkpoint_path):
            print(f"Using cached AdaCLIP checkpoint: {checkpoint_path}")
            return checkpoint_path

        print(f"Downloading AdaCLIP checkpoint to {checkpoint_path}")
        try:
            torch.hub.download_url_to_file(self.checkpoint_url, checkpoint_path, progress=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to download AdaCLIP checkpoint from {self.checkpoint_url}") from exc
        return checkpoint_path

    def _build_adaclip_trainer(self, repo_path, checkpoint_path):
        self._ensure_tokenizer_assets(repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            method_module = importlib.import_module("method")
            trainer_cls = getattr(method_module, "AdaCLIP_Trainer")
        except Exception as exc:
            raise ImportError(f"Failed to import AdaCLIP_Trainer from AdaCLIP_res repo: {repo_path}") from exc

        config = self._load_model_config(repo_path)
        output_layers = list(self.layers)
        try:
            trainer = trainer_cls(
                backbone=self.model_name,
                feat_list=output_layers,
                input_dim=config["vision_cfg"]["width"],
                output_dim=config["embed_dim"],
                learning_rate=0.0,
                device=self.device_name,
                image_size=self.image_size,
                prompting_depth=4,
                prompting_length=5,
                prompting_branch="VL",
                prompting_type="SD",
                use_hsf=True,
                k_clusters=20,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to build AdaCLIP_Trainer. Check the AdaCLIP_res repo, model name, "
                f"and image size. repo={repo_path}, model={self.model_name}"
            ) from exc

        try:
            trainer.load(checkpoint_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load AdaCLIP checkpoint: {checkpoint_path}") from exc
        return trainer

    def _ensure_tokenizer_assets(self, repo_path):
        bpe_path = os.path.join(repo_path, "method", "bpe_simple_vocab_16e6.txt.gz")
        if self._is_valid_gzip(bpe_path):
            return

        os.makedirs(os.path.dirname(bpe_path), exist_ok=True)
        tmp_path = bpe_path + ".download"
        print(
            "AdaCLIP tokenizer BPE is missing or invalid; "
            f"downloading OpenAI CLIP BPE to {bpe_path}"
        )
        try:
            torch.hub.download_url_to_file(_OPENAI_CLIP_BPE_URL, tmp_path, progress=True)
            if not self._is_valid_gzip(tmp_path):
                raise RuntimeError("downloaded BPE file is not a valid gzip archive")
            os.replace(tmp_path, bpe_path)
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(
                "Failed to prepare AdaCLIP tokenizer BPE. The cached AdaCLIP_res repo may "
                "contain a Git LFS pointer instead of the real gzip file. Install git-lfs "
                f"and run `git lfs pull` in {repo_path}, or manually download "
                f"{_OPENAI_CLIP_BPE_URL} to {bpe_path}."
            ) from exc

    @staticmethod
    def _is_valid_gzip(path):
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as handle:
                return handle.read(2) == b"\x1f\x8b"
        except OSError:
            return False

    def _load_model_config(self, repo_path):
        config_path = os.path.join(repo_path, "model_configs", f"{self.model_name}.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"AdaCLIP model config does not exist: {config_path}")
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _infer_channels(self, repo_path):
        config = self._load_model_config(repo_path)
        if self.return_projected:
            return int(config["embed_dim"])
        return int(config["vision_cfg"]["width"])

    def train(self, mode=True):
        super().train(mode)
        self.trainer.eval()
        self.model.eval()
        return self

    def _normalize_for_adaclip(self, images):
        images = images * self.imagenet_std + self.imagenet_mean
        return (images - self.clip_mean) / self.clip_std

    def _ordered_features(self, features):
        if not isinstance(features, dict):
            raise TypeError(
                "AdaCLIP extract_prompted_features must return a dict keyed by layer name, "
                f"got {type(features).__name__}."
            )
        ordered = []
        for layer in self.layers:
            key = f"layer{layer}"
            if key not in features:
                raise KeyError(f"AdaCLIP prompted features are missing key {key}. Available keys: {list(features.keys())}")
            feature = features[key]
            if feature.dim() != 4:
                raise ValueError(f"{key} must have shape [B,C,H,W], got {tuple(feature.shape)}")
            ordered.append(feature)
        return ordered

    def _debug_shapes(self, features):
        if self._debug_printed:
            return
        print("feature_backbone = adaclip_prompted")
        print(f"adaclip_model = {self.model_name}")
        print(f"clip_layers = {list(self.layers)}")
        print(f"return_projected = {self.return_projected}")
        for layer, feature in zip(self.layers, features):
            print(f"layer{layer} shape = {tuple(feature.shape)}")
        print(f"checkpoint = {self.checkpoint_path}")
        self._debug_printed = True

    def forward(self, images):
        images = self._normalize_for_adaclip(images)
        with torch.no_grad():
            if images.is_cuda:
                with torch.cuda.amp.autocast():
                    features = self.model.extract_prompted_features(
                        images,
                        layers=list(self.layers),
                        return_projected=self.return_projected,
                    )
            else:
                features = self.model.extract_prompted_features(
                    images,
                    layers=list(self.layers),
                    return_projected=self.return_projected,
                )
        features = self._ordered_features(features)
        self._debug_shapes(features)
        return features

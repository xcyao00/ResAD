from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SoftCodebookAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        num_embeddings: int = 512,
        tau: float = 0.2,
        gamma: float = 0.03,
        warmup_epochs: int = 5,
        conf_gate: bool = False,
        gate_threshold: float = 0.0,
        gate_temp: float = 0.05,
    ):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, channels)
        self.tau = tau
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.conf_gate = conf_gate
        self.gate_threshold = gate_threshold
        self.gate_temp = gate_temp
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor, epoch: Optional[int] = None) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"SoftCodebookAdapter expects [B, C, H, W], got {tuple(x.shape)}.")

        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        out_flat = self.forward_flat(x_flat, epoch=epoch)
        return out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward_flat(self, x_flat: Tensor, epoch: Optional[int] = None) -> Tensor:
        x_norm = F.normalize(x_flat, p=2, dim=1)
        codebook = self.codebook.weight
        code_norm = F.normalize(codebook, p=2, dim=1)
        sim = x_norm @ code_norm.t()

        tau = max(float(self.tau), 1e-6)
        attn = torch.softmax(sim / tau, dim=1)
        soft_code = attn @ codebook

        gamma_eff = self._gamma_eff(epoch)
        if self.conf_gate:
            s_max = sim.max(dim=1, keepdim=True).values
            gate_temp = max(float(self.gate_temp), 1e-6)
            gate = torch.sigmoid((s_max - self.gate_threshold) / gate_temp)
            gamma_eff = gamma_eff * gate

        out_flat = x_flat + gamma_eff * (soft_code - x_flat)
        return out_flat

    def _gamma_eff(self, epoch: Optional[int]) -> float:
        if epoch is None or self.warmup_epochs <= 0:
            return float(self.gamma)
        # With warmup enabled, epoch 0 intentionally starts from gamma_eff=0.
        return float(self.gamma) * min(1.0, float(epoch) / float(self.warmup_epochs))


class SoftCodebookAdapterList(nn.Module):
    def __init__(
        self,
        channels_list,
        num_embeddings: int = 512,
        tau: float = 0.2,
        gamma: float = 0.03,
        warmup_epochs: int = 5,
        conf_gate: bool = False,
        gate_threshold: float = 0.0,
        gate_temp: float = 0.05,
    ):
        super().__init__()
        self.adapters = nn.ModuleList([
            SoftCodebookAdapter(
                channels=channels,
                num_embeddings=num_embeddings,
                tau=tau,
                gamma=gamma,
                warmup_epochs=warmup_epochs,
                conf_gate=conf_gate,
                gate_threshold=gate_threshold,
                gate_temp=gate_temp,
            )
            for channels in channels_list
        ])

    def forward(self, features: List[Tensor], epoch: Optional[int] = None) -> List[Tensor]:
        if len(features) != len(self.adapters):
            raise ValueError(f"Expected {len(self.adapters)} feature levels, got {len(features)}.")
        return [adapter(feature, epoch=epoch) for adapter, feature in zip(self.adapters, features)]

    def forward_level_flat(self, level: int, x_flat: Tensor, epoch: Optional[int] = None) -> Tensor:
        return self.adapters[level].forward_flat(x_flat, epoch=epoch)


def apply_soft_codebook_if_enabled(
    args,
    soft_codebook: Optional[SoftCodebookAdapterList],
    features: List[Tensor],
    epoch: Optional[int] = None,
    debug_shapes: bool = False,
    prefix: str = "soft_codebook",
) -> List[Tensor]:
    if not getattr(args, "use_soft_codebook", False) or soft_codebook is None:
        return features

    out = soft_codebook(features, epoch=epoch)
    if debug_shapes:
        shapes = [tuple(feature.shape) for feature in out]
        print(f"[{prefix}] output shapes: {shapes}")
    warn_nonfinite_features(out, prefix=prefix)
    return out


def apply_soft_codebook_flat_if_enabled(
    args,
    soft_codebook: Optional[SoftCodebookAdapterList],
    level: int,
    x_flat: Tensor,
    epoch: Optional[int] = None,
    prefix: str = "soft_codebook",
) -> Tensor:
    if not getattr(args, "use_soft_codebook", False) or soft_codebook is None:
        return x_flat

    out = soft_codebook.forward_level_flat(level, x_flat, epoch=epoch)
    warn_nonfinite_tensor(out, prefix=f"{prefix} level {level}")
    return out


def warn_nonfinite_features(features: List[Tensor], prefix: str = "soft_codebook") -> None:
    for level, feature in enumerate(features):
        warn_nonfinite_tensor(feature, prefix=f"{prefix} level {level}")


def warn_nonfinite_tensor(tensor: Tensor, prefix: str = "soft_codebook") -> None:
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    if has_nan.item() or has_inf.item():
        print(f"[WARN] {prefix}: NaN={has_nan.item()} Inf={has_inf.item()}")

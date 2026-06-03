from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F


def apply_residual_wavelet_filter(rfeatures: List[Tensor], wave: str = "haar",
                                  hf_weight: float = 1.0,
                                  wav_mode: str = "ll_hf",
                                  ll_skip_alpha: float = 0.5,
                                  hf_gate_beta: float = 1.0,
                                  hf_skip_alpha: float = 0.75,
                                  wav_hf_normalize: bool = False) -> List[Tensor]:
    """
    Apply a 1-level Haar DWT filter in residual space.

    Each input/output feature keeps shape (B, C, H, W).
    """
    if wave != "haar":
        raise ValueError(f"Only Haar wavelet is supported, but got {wave}.")
    return [
        _apply_haar_residual_filter(
            rfeature,
            hf_weight=hf_weight,
            wav_mode=wav_mode,
            ll_skip_alpha=ll_skip_alpha,
            hf_gate_beta=hf_gate_beta,
            hf_skip_alpha=hf_skip_alpha,
            wav_hf_normalize=wav_hf_normalize,
        )
        for rfeature in rfeatures
    ]


def apply_feature_wavelet_filter(features: List[Tensor], wave: str = "haar",
                                 feature_wav_mode: str = "ll_only",
                                 hf_weight: float = 1.0,
                                 ll_skip_alpha: float = 0.5,
                                 hf_skip_alpha: float = 0.75,
                                 wav_hf_normalize: bool = False) -> List[Tensor]:
    """
    Apply a 1-level Haar DWT filter before reference matching.

    Each input/output feature keeps shape (B, C, H, W).
    """
    if wave != "haar":
        raise ValueError(f"Only Haar wavelet is supported, but got {wave}.")
    return [
        _apply_haar_feature_filter(
            feature,
            feature_wav_mode=feature_wav_mode,
            hf_weight=hf_weight,
            ll_skip_alpha=ll_skip_alpha,
            hf_skip_alpha=hf_skip_alpha,
            wav_hf_normalize=wav_hf_normalize,
        )
        for feature in features
    ]


def _apply_haar_feature_filter(x: Tensor,
                               feature_wav_mode: str = "ll_only",
                               hf_weight: float = 1.0,
                               ll_skip_alpha: float = 0.5,
                               hf_skip_alpha: float = 0.75,
                               wav_hf_normalize: bool = False) -> Tensor:
    if feature_wav_mode not in {"ll_only", "hf_only", "ll_hf", "skip_ll", "skip_hf"}:
        raise ValueError(f"Unsupported feature_wav_mode: {feature_wav_mode}.")
    ll, hf_energy = _haar_ll_hf(x)
    if feature_wav_mode == "ll_only":
        return ll
    if wav_hf_normalize:
        hf_energy = _match_hf_energy_abs_scale(hf_energy, x)
    if feature_wav_mode == "hf_only":
        return hf_energy
    if feature_wav_mode == "ll_hf":
        return ll + hf_weight * hf_energy
    if feature_wav_mode == "skip_ll":
        return ll_skip_alpha * x + (1.0 - ll_skip_alpha) * ll
    return hf_skip_alpha * x + (1.0 - hf_skip_alpha) * hf_energy


def _apply_haar_residual_filter(x: Tensor, hf_weight: float = 1.0,
                                wav_mode: str = "ll_hf",
                                ll_skip_alpha: float = 0.5,
                                hf_gate_beta: float = 1.0,
                                hf_skip_alpha: float = 0.75,
                                wav_hf_normalize: bool = False) -> Tensor:
    if wav_mode not in {"ll_hf", "ll_only", "skip_ll", "skip_hf", "hf_gate"}:
        raise ValueError(f"Unsupported wav_mode: {wav_mode}.")

    ll, hf_energy = _haar_ll_hf(x)

    if wav_mode == "ll_hf":
        return ll + hf_weight * hf_energy
    if wav_mode == "ll_only":
        return ll
    if wav_mode == "skip_ll":
        return ll_skip_alpha * x + (1.0 - ll_skip_alpha) * ll
    if wav_mode == "skip_hf":
        if wav_hf_normalize:
            hf_energy = _match_hf_energy_scale(hf_energy, x)
        return hf_skip_alpha * x + (1.0 - hf_skip_alpha) * hf_energy

    hf_norm = _normalize_hf_energy(hf_energy)
    gate = torch.sigmoid(hf_norm)
    return ll * (1.0 + hf_gate_beta * gate)


def _haar_ll_hf(x: Tensor):
    if x.dim() != 4:
        raise ValueError(f"Feature must be 4D (B, C, H, W), but got {tuple(x.shape)}.")
    B, C, H, W = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        x_dwt = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        x_dwt = x

    kernels = _haar_kernels(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)
    coeffs = F.conv2d(x_dwt, kernels, stride=2, groups=C)
    h_dwt, w_dwt = coeffs.shape[-2:]
    coeffs = coeffs.view(B, C, 4, h_dwt, w_dwt).permute(0, 2, 1, 3, 4)
    ll, lh, hl, hh = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]

    hf_energy = torch.sqrt(lh.pow(2) + hl.pow(2) + hh.pow(2) + 1e-12)
    ll = F.interpolate(ll, size=(H, W), mode="bilinear", align_corners=False)
    hf_energy = F.interpolate(hf_energy, size=(H, W), mode="bilinear", align_corners=False)
    return ll, hf_energy


def _match_hf_energy_scale(hf_energy: Tensor, x: Tensor) -> Tensor:
    hf_mean = hf_energy.mean(dim=(-2, -1), keepdim=True)
    x_mean = x.mean(dim=(-2, -1), keepdim=True)
    return hf_energy / (hf_mean + 1e-6) * x_mean


def _match_hf_energy_abs_scale(hf_energy: Tensor, x: Tensor) -> Tensor:
    hf_mean = hf_energy.mean(dim=(-2, -1), keepdim=True)
    x_abs_mean = x.abs().mean(dim=(-2, -1), keepdim=True)
    return hf_energy / (hf_mean + 1e-6) * x_abs_mean


def _normalize_hf_energy(hf_energy: Tensor) -> Tensor:
    mean = hf_energy.mean(dim=(-2, -1), keepdim=True)
    std = hf_energy.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
    return (hf_energy - mean) / std


def _haar_kernels(device, dtype) -> Tensor:
    return torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],      # LL
            [[-0.5, 0.5], [-0.5, 0.5]],    # LH
            [[-0.5, -0.5], [0.5, 0.5]],    # HL
            [[0.5, -0.5], [-0.5, 0.5]],    # HH
        ],
        device=device,
        dtype=dtype,
    ).view(4, 1, 2, 2)


def residual_wavelet_shape_test(device: str = "cpu") -> None:
    rfeatures = [
        torch.randn(2, 8, 56, 56, device=device),
        torch.randn(2, 16, 28, 28, device=device),
        torch.randn(2, 32, 15, 17, device=device),
    ]
    for wav_mode in ("ll_hf", "ll_only", "skip_ll", "skip_hf", "hf_gate"):
        filtered = apply_residual_wavelet_filter(
            rfeatures,
            wave="haar",
            hf_weight=1.0,
            wav_mode=wav_mode,
            ll_skip_alpha=0.5,
            hf_gate_beta=1.0,
            hf_skip_alpha=0.75,
            wav_hf_normalize=(wav_mode == "skip_hf"),
        )
        for before, after in zip(rfeatures, filtered):
            if before.shape != after.shape:
                raise AssertionError(
                    f"{wav_mode} changed shape from {tuple(before.shape)} to {tuple(after.shape)}."
                )

    for feature_wav_mode in ("ll_only", "hf_only", "ll_hf", "skip_ll", "skip_hf"):
        filtered = apply_feature_wavelet_filter(
            rfeatures,
            wave="haar",
            feature_wav_mode=feature_wav_mode,
            hf_weight=1.0,
            ll_skip_alpha=0.5,
            hf_skip_alpha=0.75,
            wav_hf_normalize=(feature_wav_mode in {"hf_only", "skip_hf"}),
        )
        for before, after in zip(rfeatures, filtered):
            if before.shape != after.shape:
                raise AssertionError(
                    f"{feature_wav_mode} changed shape from {tuple(before.shape)} to {tuple(after.shape)}."
                )

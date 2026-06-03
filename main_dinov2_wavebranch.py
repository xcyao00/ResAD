import argparse
import math
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.focal_loss import FocalLoss
from losses.loss import calculate_log_barrier_bg_spp_loss, calculate_log_barrier_bi_occ_loss
from losses.utils import get_logp_a, get_normal_boundary
from models.dinov2_backbone import DINOv2BackboneWrapper, DINOV2_BACKBONES, DINOV2_FEATURE_MODES
from models.dinov2_backbone import dinov2_shape_test, print_dinov2_config
from models.utils import get_logp, log_theta


warnings.filterwarnings("ignore")

TOTAL_SHOT = 4
FIRST_STAGE_EPOCH = 10
BRANCH_NAMES = ("residual", "ll", "hf")
logp_wrapper = log_theta


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_settings():
    from classes import CAPSULES_TO_CAPSULES
    from classes import MVTEC_TO_BRATS, MVTEC_TO_BTAD, MVTEC_TO_MPDD, MVTEC_TO_MVTEC
    from classes import MVTEC_TO_MVTEC3D, MVTEC_TO_MVTECLOCO, MVTEC_TO_VISA
    from classes import VISA_TO_MVTEC, VISA_TO_VISA

    return {
        "visa_to_mvtec": VISA_TO_MVTEC,
        "mvtec_to_visa": MVTEC_TO_VISA,
        "mvtec_to_btad": MVTEC_TO_BTAD,
        "mvtec_to_mvtec3d": MVTEC_TO_MVTEC3D,
        "mvtec_to_mpdd": MVTEC_TO_MPDD,
        "mvtec_to_mvtecloco": MVTEC_TO_MVTECLOCO,
        "mvtec_to_brats": MVTEC_TO_BRATS,
        "mvtec_to_mvtec": MVTEC_TO_MVTEC,
        "visa_to_visa": VISA_TO_VISA,
        "capsules_to_capsules": CAPSULES_TO_CAPSULES,
    }


class SingleScaleConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def active_branches(args):
    if args.branch_mode == "res_only":
        return ("residual",)
    if args.branch_mode == "res_ll_hf":
        return BRANCH_NAMES
    raise ValueError(f"Unsupported branch_mode: {args.branch_mode}")


def branch_loss_weight(args, branch_name):
    if branch_name == "residual":
        return args.branch_loss_w_residual
    if branch_name == "ll":
        return args.branch_loss_w_ll
    if branch_name == "hf":
        return args.branch_loss_w_hf
    raise ValueError(f"Unsupported branch name: {branch_name}")


def branch_score_weight(args, branch_name):
    if branch_name == "residual":
        return args.score_w_residual
    if branch_name == "ll":
        return args.score_w_ll
    if branch_name == "hf":
        return args.score_w_hf
    raise ValueError(f"Unsupported branch name: {branch_name}")


def build_encoder(args):
    if args.backbone not in DINOV2_BACKBONES:
        raise ValueError(f"main_dinov2_wavebranch.py supports DINOv2 only, got {args.backbone}.")
    encoder = DINOv2BackboneWrapper(
        model_name=args.backbone,
        freeze=True,
        feature_mode=args.dinov2_feature_mode,
        layers=args.dinov2_layers,
        proj_dim=args.dinov2_proj_dim,
    ).to(args.device)
    encoder.eval()
    feat_dims = encoder.feature_info.channels()
    if len(feat_dims) != 1:
        raise ValueError(
            "Wavebranch experiment expects one DINOv2 final feature level. "
            f"Got feature dims {feat_dims}; use --dinov2_feature_mode final_only."
        )
    print_dinov2_config(encoder, image_size=224)
    return encoder, feat_dims[0]


def build_train_loaders(args, classes):
    from datasets.capsules import CAPSULES, CAPSULESANO
    from datasets.mvtec import MVTEC, MVTECANO
    from datasets.visa import VISA, VISAANO

    if args.classes == "capsules":
        train_dataset1 = CAPSULES(
            args.train_dataset_dir,
            class_name=classes["seen"],
            train=True,
            normalize="w50",
            img_size=224,
            crp_size=224,
            msk_size=224,
            msk_crp_size=224,
        )
        train_dataset2 = CAPSULESANO(
            args.train_dataset_dir,
            class_name=classes["seen"],
            train=True,
            normalize="w50",
            img_size=224,
            crp_size=224,
            msk_size=224,
            msk_crp_size=224,
        )
    elif classes["seen"][0] in MVTEC.CLASS_NAMES:
        train_dataset1 = MVTEC(
            args.train_dataset_dir,
            class_name=classes["seen"],
            train=True,
            normalize="w50",
            img_size=224,
            crp_size=224,
            msk_size=224,
            msk_crp_size=224,
        )
        train_dataset2 = MVTECANO(
            args.train_dataset_dir,
            class_name=classes["seen"],
            train=True,
            normalize="w50",
            img_size=224,
            crp_size=224,
            msk_size=224,
            msk_crp_size=224,
        )
    else:
        train_dataset1 = VISA(
            args.train_dataset_dir,
            class_name=classes["seen"],
            train=True,
            normalize="w50",
            img_size=224,
            crp_size=224,
            msk_size=224,
            msk_crp_size=224,
        )
        train_dataset2 = VISAANO(
            args.train_dataset_dir,
            class_name=classes["seen"],
            train=True,
            normalize="w50",
            img_size=224,
            crp_size=224,
            msk_size=224,
            msk_crp_size=224,
        )

    train_loader1 = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    return train_loader1, train_loader2


def build_test_dataset(args, class_name):
    from datasets.brats import BRATS
    from datasets.btad import BTAD
    from datasets.capsules import CAPSULES
    from datasets.mpdd import MPDD
    from datasets.mvtec import MVTEC
    from datasets.mvtec_3d import MVTEC3D
    from datasets.mvtec_loco import MVTECLOCO
    from datasets.visa import VISA

    kwargs = dict(train=False, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
    if args.classes == "capsules":
        return CAPSULES(args.test_dataset_dir, class_name=class_name, **kwargs)
    if class_name in MVTEC.CLASS_NAMES:
        return MVTEC(args.test_dataset_dir, class_name=class_name, **kwargs)
    if class_name in VISA.CLASS_NAMES:
        return VISA(args.test_dataset_dir, class_name=class_name, **kwargs)
    if class_name in BTAD.CLASS_NAMES:
        return BTAD(args.test_dataset_dir, class_name=class_name, **kwargs)
    if class_name in MVTEC3D.CLASS_NAMES:
        return MVTEC3D(args.test_dataset_dir, class_name=class_name, **kwargs)
    if class_name in MPDD.CLASS_NAMES:
        return MPDD(args.test_dataset_dir, class_name=class_name, **kwargs)
    if class_name in MVTECLOCO.CLASS_NAMES:
        return MVTECLOCO(args.test_dataset_dir, class_name=class_name, **kwargs)
    if class_name in BRATS.CLASS_NAMES:
        return BRATS(args.test_dataset_dir, class_name=class_name, **kwargs)
    raise ValueError(f"Unrecognized class name: {class_name}")


def _haar_kernels(device, dtype) -> Tensor:
    return torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, -0.5], [0.5, 0.5]],
            [[0.5, -0.5], [-0.5, 0.5]],
        ],
        device=device,
        dtype=dtype,
    ).view(4, 1, 2, 2)


def haar_ll_hf(residual: Tensor, hf_normalize: bool = True, wave: str = "haar"):
    if wave != "haar":
        raise ValueError(f"Only Haar wavelet is supported, got {wave}.")
    if residual.dim() != 4:
        raise ValueError(f"Expected residual [B,C,H,W], got {tuple(residual.shape)}.")

    b, c, h, w = residual.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x_dwt = F.pad(residual, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        x_dwt = residual

    kernels = _haar_kernels(device=residual.device, dtype=residual.dtype).repeat(c, 1, 1, 1)
    coeffs = F.conv2d(x_dwt, kernels, stride=2, groups=c)
    h_dwt, w_dwt = coeffs.shape[-2:]
    coeffs = coeffs.view(b, c, 4, h_dwt, w_dwt).permute(0, 2, 1, 3, 4)
    ll, lh, hl, hh = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]

    hf = torch.sqrt(lh.pow(2) + hl.pow(2) + hh.pow(2) + 1e-12)
    ll = F.interpolate(ll, size=(h, w), mode="bilinear", align_corners=False)
    hf = F.interpolate(hf, size=(h, w), mode="bilinear", align_corners=False)

    if hf_normalize:
        hf_mean = hf.mean(dim=(-2, -1), keepdim=True)
        residual_mean = residual.mean(dim=(-2, -1), keepdim=True)
        hf = hf / (hf_mean + 1e-6) * residual_mean
    return ll, hf


def make_branch_features(args, residual: Tensor):
    if args.branch_mode == "res_only":
        return {"residual": residual}
    ll, hf = haar_ll_hf(residual, hf_normalize=args.hf_normalize, wave=args.wave)
    return {"residual": residual, "ll": ll, "hf": hf}


def wavebranch_shape_test(device="cpu"):
    residual = torch.randn(2, 384, 16, 16, device=device)
    class _Args:
        branch_mode = "res_ll_hf"
        hf_normalize = True
        wave = "haar"

    branches = make_branch_features(_Args(), residual)
    for name, feature in branches.items():
        if tuple(feature.shape) != (2, 384, 16, 16):
            raise AssertionError(f"{name} branch shape mismatch: {tuple(feature.shape)}")
    constraintor = SingleScaleConv(384).to(device).eval()
    with torch.no_grad():
        out = constraintor(branches["residual"])
    if tuple(out.shape) != tuple(residual.shape):
        raise AssertionError(f"constraintor changed shape: {tuple(out.shape)}")


def load_test_reference_features(root_dir, class_names, device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, "layer1.npy"))).to(device)
        k1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        refs[class_name] = (layer1_refs[:k1, :],)
    return refs


def build_branch_models(args, channels):
    from utils import BoundaryAverager
    from models.fc_flow import load_flow_model

    branches = active_branches(args)
    constraintors = nn.ModuleDict({name: SingleScaleConv(channels).to(args.device) for name in branches})
    estimators = nn.ModuleDict({name: load_flow_model(args, channels).to(args.device) for name in branches})

    optimizer_constraints = {
        name: torch.optim.Adam(constraintors[name].parameters(), lr=args.lr, weight_decay=0.0005)
        for name in branches
    }
    optimizer_flows = {
        name: torch.optim.Adam(estimators[name].parameters(), lr=args.lr, weight_decay=0.0005)
        for name in branches
    }
    schedulers = []
    for name in branches:
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizer_constraints[name], milestones=[70, 90], gamma=0.1))
        schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizer_flows[name], milestones=[70, 90], gamma=0.1))
    boundary_ops = {name: BoundaryAverager(num_levels=1) for name in branches}
    return constraintors, estimators, optimizer_constraints, optimizer_flows, schedulers, boundary_ops


def train_constraintor_branch(args, feature, constraintor, optimizer, masks, loss_weight):
    target = feature.detach().clone()
    output = constraintor(feature)

    bs, dim, h, w = output.size()
    e = output.permute(0, 2, 3, 1).reshape(-1, dim)
    t = target.permute(0, 2, 3, 1).reshape(-1, dim)
    m = F.interpolate(masks, size=(h, w), mode="nearest").squeeze(1).reshape(-1)

    loss, _, _ = calculate_log_barrier_bi_occ_loss(e, m, t)
    optimizer.zero_grad()
    (loss_weight * loss).backward()
    optimizer.step()
    return output, loss.item()


def _iter_chunks(args, total, n_batch):
    perm = torch.randperm(total, device=args.device)
    for start in range(0, total, n_batch):
        end = min(start + n_batch, total)
        yield perm[start:end]


def train_nf_branch(args, feature, estimator, optimizer, masks, boundary_ops, epoch, loss_weight, n_batch):
    from models.modules import get_position_encoding

    train_loss_total, total_num = 0, 0
    bs, dim, h, w = feature.size()
    e = feature.permute(0, 2, 3, 1).reshape(-1, dim)
    masks_ = F.interpolate(masks, size=(h, w), mode="nearest").squeeze(1).reshape(-1)
    pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)

    total = bs * h * w
    for idx in _iter_chunks(args, total, n_batch):
        p_b = pos_embed[idx]
        e_b = e[idx]
        m_b = masks_[idx]

        if args.flow_arch == "flow_model":
            z, log_jac_det = estimator(e_b)
        else:
            z, log_jac_det = estimator(e_b, [p_b])

        if epoch < FIRST_STAGE_EPOCH:
            logps = get_logp(dim, z, log_jac_det)
            logps = logps / dim
            loss = -logp_wrapper(logps).mean()
            b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
            boundary_ops.update_boundary(b_n, 0)
        else:
            logps = get_logp(dim, z, log_jac_det)
            logps = logps / dim
            b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
            boundary_ops.update_boundary(b_n, 0)

            if m_b.sum() == 0:
                loss = -logp_wrapper(logps).mean()
            else:
                logps_a = get_logp_a(dim, z, log_jac_det)
                loss_ml = -logp_wrapper(logps[m_b == 0]).mean()
                loss_ml_a = -logp_wrapper(logps_a[m_b == 1]).mean()
                logits = torch.stack([logps, logps_a], dim=-1)
                s = torch.softmax(logits, dim=-1)
                loss_focal = FocalLoss()(s, m_b.unsqueeze(-1))
                b_n = boundary_ops.get_boundary(0)
                b_a = b_n - args.margin_tau
                loss_n_con, loss_a_con = calculate_log_barrier_bg_spp_loss(logps, m_b, (b_n, b_a))
                loss = loss_ml + loss_ml_a + loss_focal + args.bgspp_lambda * (loss_n_con + loss_a_con)

        if not torch.isfinite(loss.detach()).item():
            print(f"[WARN] non-finite NF loss at epoch={epoch}; skipping branch optimizer step.")
            continue
        optimizer.zero_grad()
        (loss_weight * loss).backward()
        optimizer.step()
        train_loss_total += loss.item()
        total_num += 1
    return train_loss_total, total_num


def scores_from_normal_logps(logps_list, size=224):
    logps = torch.cat(logps_list, dim=0)
    logps -= torch.max(logps)
    probs = torch.exp(logps)
    scores = F.interpolate(probs.unsqueeze(1), size=size, mode="bilinear", align_corners=True).squeeze().cpu().numpy()
    scores = scores.max() - scores
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)
    return scores


def scores_from_abnormal_probs(prob_list, size=224):
    probs = torch.cat(prob_list, dim=0)
    scores = F.interpolate(probs.unsqueeze(1), size=size, mode="bilinear", align_corners=True).squeeze().cpu().numpy()
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)
    return scores


def metric_from_scores(scores, labels, gt_masks):
    from utils import calculate_metrics

    return list(calculate_metrics(scores, labels, gt_masks, pro=False, only_max_value=True))


def validate_wavebranch(args, encoder, constraintors, estimators, test_loader, ref_features, device, class_name):
    from models.modules import get_position_encoding
    from utils import get_matched_ref_features_by_mode, get_residual_features_by_mode

    branches = active_branches(args)
    encoder.eval()
    for name in branches:
        constraintors[name].eval()
        estimators[name].eval()

    label_list, gt_mask_list = [], []
    logps1 = {name: [] for name in branches}
    logps2 = {name: [] for name in branches}

    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating {class_name}")
    for batch in test_loader:
        progress_bar.update(1)
        image, label, mask = batch[0], batch[1], batch[2]
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())
        image = image.to(device)
        size = image.shape[-1]

        with torch.no_grad():
            features = encoder(image)
            mfeatures = get_matched_ref_features_by_mode(
                features,
                ref_features,
                match_mode=args.match_mode,
                topk=args.match_topk,
                tau=args.match_tau,
                chunk_size=args.match_chunk_size,
            )
            residual = get_residual_features_by_mode(features, mfeatures, mode=args.residual_mode)[0]
            branch_inputs = make_branch_features(args, residual)

            for name in branches:
                feature = constraintors[name](branch_inputs[name])
                bs, dim, h, w = feature.size()
                e = feature.permute(0, 2, 3, 1).reshape(-1, dim)
                pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(device).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                if args.flow_arch == "flow_model":
                    z, log_jac_det = estimators[name](e)
                else:
                    z, log_jac_det = estimators[name](e, [pos_embed])
                logp = get_logp(dim, z, log_jac_det) / dim
                logps1[name].append(logp.reshape(bs, h, w))
                logp_a = get_logp_a(dim, z, log_jac_det)
                logits = torch.stack([logp, logp_a], dim=-1)
                sa = torch.softmax(logits, dim=-1)[:, 1]
                logps2[name].append(sa.reshape(bs, h, w))

    progress_bar.close()
    labels = np.concatenate(label_list)
    gt_masks = np.concatenate(gt_mask_list, axis=0)

    branch_scores = {}
    metrics = {}
    for name in branches:
        scores1 = scores_from_normal_logps(logps1[name], size=size)
        scores2 = scores_from_abnormal_probs(logps2[name], size=size)
        scores = (scores1 + scores2) / 2
        branch_scores[name] = scores
        metrics[name] = metric_from_scores(scores, labels, gt_masks)

    fused = None
    for name in branches:
        weighted = branch_score_weight(args, name) * branch_scores[name]
        fused = weighted if fused is None else fused + weighted
    metrics["fused"] = metric_from_scores(fused, labels, gt_masks)
    return metrics


def format_metrics(values):
    img_auc, img_ap, img_f1, pix_auc, pix_ap, pix_f1, pix_aupro = values
    return (
        f"Image AUC | AP | F1: {img_auc:.3f} | {img_ap:.3f} | {img_f1:.3f}, "
        f"Pixel AUC | AP | F1 | AUPRO: {pix_auc:.3f} | {pix_ap:.3f} | {pix_f1:.3f} | {pix_aupro:.3f}"
    )


def print_branch_average(label, values):
    print(f"({label}) Average {format_metrics(values)}")


def main(args):
    args.feature_levels = 1
    settings = get_settings()
    if args.setting not in settings:
        raise ValueError(f"Dataset setting must be in {settings.keys()}, but got {args.setting}.")
    classes = settings[args.setting]

    print("[WaveBranch] branch_mode:", args.branch_mode)
    print("[WaveBranch] score weights:", args.score_w_residual, args.score_w_ll, args.score_w_hf)
    print("[WaveBranch] branch loss weights:", args.branch_loss_w_residual, args.branch_loss_w_ll, args.branch_loss_w_hf)
    print("[WaveBranch] hf_normalize:", args.hf_normalize)

    train_loader1, train_loader2 = build_train_loaders(args, classes)
    encoder, channels = build_encoder(args)
    constraintors, estimators, optim_c, optim_nf, schedulers, boundary_ops = build_branch_models(args, channels)
    from utils import get_mc_matched_ref_features_by_mode, get_mc_reference_features
    from utils import get_residual_features_by_mode

    best_value = -float("inf")
    for epoch in range(args.epochs):
        for name in active_branches(args):
            constraintors[name].train()
            estimators[name].train()

        train_loader = train_loader1 if epoch < FIRST_STAGE_EPOCH else train_loader2
        train_loss_total, total_num = 0.0, 0
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
        for batch in train_loader:
            progress_bar.update(1)
            images, _, masks, class_names = batch
            images = images.to(args.device)
            masks = masks.to(args.device)

            with torch.no_grad():
                features = encoder(images)
                ref_features = get_mc_reference_features(
                    encoder,
                    args.train_dataset_dir,
                    class_names,
                    images.device,
                    args.train_ref_shot,
                )
                mfeatures = get_mc_matched_ref_features_by_mode(
                    features,
                    class_names,
                    ref_features,
                    match_mode=args.match_mode,
                    topk=args.match_topk,
                    tau=args.match_tau,
                    chunk_size=args.match_chunk_size,
                )
                residual = get_residual_features_by_mode(features, mfeatures, mode=args.residual_mode)[0]
                branch_inputs = make_branch_features(args, residual)

            for name in active_branches(args):
                loss_weight = branch_loss_weight(args, name)
                constrained, loss_c = train_constraintor_branch(
                    args,
                    branch_inputs[name],
                    constraintors[name],
                    optim_c[name],
                    masks,
                    loss_weight,
                )
                train_loss_total += loss_c
                total_num += 1
                loss_nf, num_nf = train_nf_branch(
                    args,
                    constrained.detach().clone(),
                    estimators[name],
                    optim_nf[name],
                    masks,
                    boundary_ops[name],
                    epoch,
                    loss_weight,
                    args.n_batch,
                )
                train_loss_total += loss_nf
                total_num += num_nf

        for scheduler in schedulers:
            scheduler.step()
        progress_bar.close()
        print(f"Epoch[{epoch}/{args.epochs}]: train_loss: {train_loss_total / max(total_num, 1)}")

        if (epoch + 1) % args.eval_freq != 0:
            continue

        test_ref_features = load_test_reference_features(
            args.test_ref_feature_dir,
            classes["unseen"],
            args.device,
            args.num_ref_shot,
        )
        metric_lists = {name: [] for name in active_branches(args)}
        metric_lists["fused"] = []
        for class_name in classes["unseen"]:
            test_dataset = build_test_dataset(args, class_name)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
            metrics = validate_wavebranch(
                args,
                encoder,
                constraintors,
                estimators,
                test_loader,
                test_ref_features[class_name],
                args.device,
                class_name,
            )
            for name, values in metrics.items():
                metric_lists[name].append(values)
            print(f"Epoch: {epoch}, Class Name: {class_name}, (Fused) {format_metrics(metrics['fused'])}")

        average_metrics = {name: np.mean(np.asarray(values), axis=0) for name, values in metric_lists.items()}
        if "residual" in average_metrics:
            print_branch_average("Residual", average_metrics["residual"])
        if "ll" in average_metrics:
            print_branch_average("LL", average_metrics["ll"])
        if "hf" in average_metrics:
            print_branch_average("HF", average_metrics["hf"])
        print_branch_average("Fused", average_metrics["fused"])

        fused_metrics = average_metrics["fused"]
        current_value = fused_metrics[0] if args.save_metric == "fused_image_auc" else fused_metrics[4]
        if current_value > best_value:
            best_value = current_value
            os.makedirs(args.checkpoint_path, exist_ok=True)
            state_dict = {"score_weights": {
                "residual": args.score_w_residual,
                "ll": args.score_w_ll,
                "hf": args.score_w_hf,
            }}
            for name in active_branches(args):
                state_dict[f"constraintor_{name}"] = constraintors[name].state_dict()
                state_dict[f"estimator_{name}"] = estimators[name].state_dict()
            torch.save(state_dict, os.path.join(args.checkpoint_path, f"{args.setting}_epoch_{epoch}_wavebranch.pth"))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="")
    parser.add_argument("--setting", type=str, default="visa_to_mvtec")
    parser.add_argument("--classes", type=str, default="none")
    parser.add_argument("--train_dataset_dir", type=str, default="")
    parser.add_argument("--test_dataset_dir", type=str, default="")
    parser.add_argument("--test_ref_feature_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/")
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--backbone", type=str, default="dinov2_vits14", choices=DINOV2_BACKBONES)
    parser.add_argument("--dinov2_feature_mode", type=str, default="final_only", choices=DINOV2_FEATURE_MODES)
    parser.add_argument("--dinov2_layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--dinov2_proj_dim", type=int, default=0)
    parser.add_argument("--dino_shape_test", action="store_true")
    parser.add_argument("--wavebranch_shape_test", action="store_true")

    parser.add_argument("--flow_arch", type=str, default="conditional_flow_model")
    parser.add_argument("--feature_levels", default=1, type=int)
    parser.add_argument("--coupling_layers", type=int, default=10)
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_embed_dim", type=int, default=256)
    parser.add_argument("--pos_beta", type=float, default=0.05)
    parser.add_argument("--margin_tau", type=float, default=0.1)
    parser.add_argument("--bgspp_lambda", type=float, default=1.0)
    parser.add_argument("--train_ref_shot", type=int, default=4)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--n_batch", type=int, default=8192)

    parser.add_argument("--match_mode", type=str, default="hard", choices=["hard", "soft_topk"])
    parser.add_argument("--match_topk", type=int, default=5)
    parser.add_argument("--match_tau", type=float, default=0.1)
    parser.add_argument("--match_chunk_size", type=int, default=2048)
    parser.add_argument("--residual_mode", type=str, default="sq", choices=["sq", "abs", "signed"])
    parser.add_argument("--wave", type=str, default="haar", choices=["haar"])
    parser.add_argument("--branch_mode", type=str, default="res_ll_hf", choices=["res_only", "res_ll_hf"])
    parser.add_argument("--score_w_residual", type=float, default=0.5)
    parser.add_argument("--score_w_ll", type=float, default=0.25)
    parser.add_argument("--score_w_hf", type=float, default=0.25)
    parser.add_argument("--branch_loss_w_residual", type=float, default=1.0)
    parser.add_argument("--branch_loss_w_ll", type=float, default=1.0)
    parser.add_argument("--branch_loss_w_hf", type=float, default=1.0)
    parser.add_argument("--hf_normalize", action="store_true", default=True)
    parser.add_argument("--no_hf_normalize", dest="hf_normalize", action="store_false")
    parser.add_argument("--save_metric", type=str, default="fused_image_auc", choices=["fused_image_auc", "fused_pixel_ap"])
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    init_seeds(42)
    if args.wavebranch_shape_test:
        test_device = args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
        wavebranch_shape_test(device=test_device)
        print("Wavebranch shape test passed.")
        raise SystemExit(0)
    if args.dino_shape_test:
        test_device = args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
        dinov2_shape_test(
            model_name=args.backbone,
            device=test_device,
            feature_mode=args.dinov2_feature_mode,
            layers=args.dinov2_layers,
            proj_dim=args.dinov2_proj_dim,
        )
        print("DINOv2 shape test passed.")
        raise SystemExit(0)
    main(args)

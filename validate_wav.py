import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

from models.modules import get_position_encoding
from models.utils import get_logp
from utils import get_residual_features_by_mode, get_matched_ref_features_by_mode
from utils import calculate_metrics
from residual_wavelet import apply_feature_wavelet_filter, apply_residual_wavelet_filter
from models.soft_codebook import apply_soft_codebook_flat_if_enabled
from raw_vqops import apply_raw_vqops_if_enabled
from losses.utils import get_logp_a

warnings.filterwarnings('ignore')


def apply_feature_wavelet_from_args(args, features):
    return apply_feature_wavelet_filter(
        features,
        wave=args.wave,
        feature_wav_mode=args.feature_wav_mode,
        hf_weight=args.hf_weight,
        ll_skip_alpha=args.ll_skip_alpha,
        hf_skip_alpha=getattr(args, "hf_skip_alpha", 0.75),
        wav_hf_normalize=getattr(args, "wav_hf_normalize", False),
    )


def assert_feature_shapes_match(features, matched_features, prefix):
    for level, (feature, matched) in enumerate(zip(features, matched_features)):
        if feature.shape != matched.shape:
            raise ValueError(
                f"{prefix} level {level} shape mismatch: "
                f"feature={tuple(feature.shape)}, matched={tuple(matched.shape)}"
            )


def validate(args, encoder, constraintor, soft_codebook, raw_vq_ops, estimators, test_loader, ref_features, device, class_name, epoch=None):
    if raw_vq_ops is not None:
        raw_vq_ops.eval()
    constraintor.eval()
    if soft_codebook is not None:
        soft_codebook.eval()
    for estimator in estimators:  
        estimator.eval()
    
    label_list, gt_mask_list = [], []
    logps1_list = [list() for _ in range(args.feature_levels)]
    logps2_list = [list() for _ in range(args.feature_levels)]
    
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating {class_name}")
    
    for idx, batch in enumerate(test_loader):
        progress_bar.update(1)
        
        image, label, mask = batch[0], batch[1], batch[2]  
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())
        
        image = image.to(device)
        size = image.shape[-1]
        
        with torch.no_grad():
            features = encoder(image)
            if args.use_wav and args.wav_on == 'feature':
                features = apply_feature_wavelet_from_args(args, features)
            mfeatures = get_matched_ref_features_by_mode(
                features,
                ref_features,
                match_mode=args.match_mode,
                topk=args.match_topk,
                tau=args.match_tau,
                chunk_size=args.match_chunk_size,
            )
            assert_feature_shapes_match(features, mfeatures, "validate matching")
            rfeatures = get_residual_features_by_mode(
                features,
                mfeatures,
                mode=args.residual_mode,
            )
            if args.use_wav and args.wav_on == 'residual':
                rfeatures = apply_residual_wavelet_filter(
                    rfeatures,
                    wave=args.wave,
                    hf_weight=args.hf_weight,
                    wav_mode=args.wav_mode,
                    ll_skip_alpha=args.ll_skip_alpha,
                    hf_gate_beta=args.hf_gate_beta,
                    hf_skip_alpha=getattr(args, "hf_skip_alpha", 0.75),
                    wav_hf_normalize=getattr(args, "wav_hf_normalize", False),
                )

            if getattr(args, "use_raw_vqops", False) and args.raw_vq_pos == "pre_constraintor":
                rfeatures = apply_raw_vqops_if_enabled(
                    args,
                    raw_vq_ops,
                    rfeatures,
                    prefix="validate_pre_constraintor",
                )
            
            rfeatures = constraintor(*rfeatures)

            if getattr(args, "use_raw_vqops", False) and args.raw_vq_pos == "post_constraintor":
                rfeatures = apply_raw_vqops_if_enabled(
                    args,
                    raw_vq_ops,
                    rfeatures,
                    prefix="validate_post_constraintor",
                )
        
            for l in range(args.feature_levels):
                e = rfeatures[l]  
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                e = apply_soft_codebook_flat_if_enabled(
                    args,
                    soft_codebook,
                    l,
                    e,
                    epoch=epoch,
                    prefix="validate_soft_codebook",
                )
                
                pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                estimator = estimators[l]

                if args.flow_arch == 'flow_model':
                    z, log_jac_det = estimator(e)  
                else:
                    z, log_jac_det = estimator(e, [pos_embed, ])

                logps = get_logp(dim, z, log_jac_det)  
                logps = logps / dim  
                logps1_list[l].append(logps.reshape(bs, h, w))
                
                logps_a = get_logp_a(dim, z, log_jac_det)  
                logits = torch.stack([logps, logps_a], dim=-1)  
                sa = torch.softmax(logits, dim=-1)[:, 1]
                logps2_list[l].append(sa.reshape(bs, h, w))
    
    progress_bar.close()
    
    labels = np.concatenate(label_list)
    gt_masks = np.concatenate(gt_mask_list, axis=0)
    
    scores1 = convert_to_anomaly_scores(logps1_list, feature_levels=args.feature_levels, size=size)
    scores2 = aggregate_anomaly_scores(logps2_list, feature_levels=args.feature_levels, size=size)
    
    img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1 = calculate_metrics(scores1, labels, gt_masks, pro=False, only_max_value=True)
    img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2 = calculate_metrics(scores2, labels, gt_masks, pro=False, only_max_value=True)
    
    scores = (scores1 + scores2) / 2
    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(scores, labels, gt_masks, pro=False, only_max_value=True)
    
    metrics = {}
    metrics['scores1'] = [img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1]
    metrics['scores2'] = [img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2]
    metrics['scores'] = [img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro]
    
    return metrics


def load_soft_codebook_state(args, checkpoint, soft_codebook):
    if not args.use_soft_codebook or soft_codebook is None:
        return
    if 'soft_codebook' not in checkpoint:
        print("[SoftCodebook] checkpoint has no soft_codebook state; skipping load.")
        return
    soft_codebook.load_state_dict(checkpoint['soft_codebook'], strict=False)


def convert_to_anomaly_scores(logps_list, feature_levels=3, size=224):
    normal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        logps = torch.cat(logps_list[l], dim=0)  
        logps -= torch.max(logps) 
        probs = torch.exp(logps) 
        normal_map[l] = F.interpolate(probs.unsqueeze(1), size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    scores = np.zeros_like(normal_map[0])
    for l in range(feature_levels):
        scores += normal_map[l]
    scores = scores.max() - scores 
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)
    return scores


def aggregate_anomaly_scores(logps_list, feature_levels=3, size=224):
    abnormal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        probs = torch.cat(logps_list[l], dim=0)  
        abnormal_map[l] = F.interpolate(probs.unsqueeze(1), size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    scores = np.zeros_like(abnormal_map[0])
    for l in range(feature_levels):
        scores += abnormal_map[l]
    scores /= feature_levels
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)
    return scores

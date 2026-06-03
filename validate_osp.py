import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

from models.modules import get_position_encoding
from models.utils import get_logp
from utils import get_residual_features, get_matched_ref_features, get_fourier_residual_features
from utils import calculate_metrics
from losses.utils import get_logp_a

warnings.filterwarnings('ignore')

# ==========================================
# OSP 適用関数 (main_osp.pyと同じもの)
# ==========================================
def apply_osp(residuals_list, proj_matrices, means):
    osp_results = []
    for i, res in enumerate(residuals_list):
        B, C, H, W = res.shape
        res_flat = res.permute(0, 2, 3, 1).reshape(-1, C)
        res_centered = res_flat - means[i]
        res_parallel = torch.mm(res_centered, proj_matrices[i])
        res_ortho = res_centered - res_parallel
        osp_results.append(res_ortho.reshape(B, H, W, C).permute(0, 3, 1, 2))
    return osp_results
# ==========================================

def validate(args, encoder, constraintor, estimators, test_loader, ref_features, device, class_name, osp_proj_matrices, osp_means):
    constraintor.eval()
    for estimator in estimators:  
        estimator.eval()
    
    label_list, gt_mask_list = [], []
    logps1_list = [list() for _ in range(args.feature_levels)]
    logps2_list = [list() for _ in range(args.feature_levels)]
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating")
    
    for idx, batch in enumerate(test_loader):
        progress_bar.update(1)
        
        image, label, mask = batch[0], batch[1], batch[2]  
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())
        
        image = image.to(device)
        size = image.shape[-1]
        
        with torch.no_grad():
            if args.backbone in ['wide_resnet50_2', 'tf_efficientnet_b6', 'vit_base_patch14']:
                features = encoder(image)
                mfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
            else:
                features = encoder.encode_image_from_tensors(image)
                for i in range(len(features)):
                    b, l, c = features[i].shape
                    features[i] = features[i].permute(0, 2, 1).reshape(b, c, 16, 16)
                mfeatures = get_matched_ref_features(features, ref_features)
                rfeatures = get_residual_features(features, mfeatures)
            
            # --- VQとEFDMの代わりにOSPを適用 ---
            rfeatures = apply_osp(rfeatures, osp_proj_matrices, osp_means)
            
            # --- constraintorで滑らかに補正 ---
            rfeatures = constraintor(*rfeatures)
        
            for l in range(args.feature_levels):
                e = rfeatures[l]  
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                
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
    scores1 = convert_to_anomaly_scores(logps1_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    scores2 = aggregate_anomaly_scores(logps2_list, feature_levels=args.feature_levels, class_name=class_name, size=size)
    
    img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1 = calculate_metrics(scores1, labels, gt_masks, pro=False, only_max_value=True)
    img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2 = calculate_metrics(scores2, labels, gt_masks, pro=False, only_max_value=True)
    
    scores = (scores1 + scores2) / 2
    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = calculate_metrics(scores, labels, gt_masks, pro=False, only_max_value=True)
    
    metrics = {}
    metrics['scores1'] = [img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1]
    metrics['scores2'] = [img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2]
    metrics['scores'] = [img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro]
    
    return metrics


def convert_to_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
    normal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        logps = torch.cat(logps_list[l], dim=0)  
        logps-= torch.max(logps) 
        probs = torch.exp(logps) 
        normal_map[l] = F.interpolate(probs.unsqueeze(1), size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    scores = np.zeros_like(normal_map[0])
    for l in range(feature_levels):
        scores += normal_map[l]

    scores = scores.max() - scores 
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores


def aggregate_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
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

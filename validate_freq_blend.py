import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

from models.modules import get_position_encoding
from models.utils import get_logp
from utils import calculate_metrics
from losses.utils import get_logp_a

warnings.filterwarnings('ignore')

# 推論時専用の分割統治マッチング関数
def get_freq_matched_residuals_infer(test_lf_list, test_hf_list, ref_lf_list, ref_hf_list, alpha=0.5, pos_flag=True, device='cuda:0'):
    rfeatures = []
    for l in range(len(test_lf_list)):
        t_lf = test_lf_list[l]
        t_hf = test_hf_list[l]
        r_lf_all = ref_lf_list[l]
        r_hf_all = ref_hf_list[l]
        
        B, C, H, W = t_lf.shape
        t_lf_flat = t_lf.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        
        t_lf_n = F.normalize(t_lf_flat, p=2, dim=1)
        r_lf_n = F.normalize(r_lf_all, p=2, dim=1)
        
        dist = t_lf_n @ r_lf_n.T
        cidx = torch.argmax(dist, dim=1)
        
        m_lf = r_lf_all[cidx].reshape(B, H, W, C).permute(0, 3, 1, 2)
        m_hf = r_hf_all[cidx].reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        res_lf = t_lf - m_lf
        res_hf = t_hf - m_hf
        
        rfeature = alpha * res_lf + (1.0 - alpha) * res_hf
        
        if pos_flag:
            pos_embed = get_position_encoding(C, H, W).to(device).unsqueeze(0).repeat(B, 1, 1, 1)
            rfeature = rfeature + pos_embed
        
        rfeatures.append(rfeature)
    return rfeatures


def validate(args, encoder, constraintor, wav_filter, estimators, test_loader, ref_features, device, class_name):
    constraintor.eval()
    for estimator in estimators:  
        estimator.eval()
        
    ref_lf_cat = []
    ref_hf_cat = []
    
    with torch.no_grad():
        dummy_img = torch.zeros(1, 3, 224, 224).to(device)
        dummy_feats = encoder(dummy_img)
        
        for l in range(args.feature_levels):
            _, C, H, W = dummy_feats[l].shape
            
            K = ref_features[l].shape[0] // (H * W)
            ref_4d = ref_features[l].view(K, H, W, C).permute(0, 3, 1, 2)
            
            ref_lf, ref_hf = wav_filter.get_LF_HF(ref_4d)
            
            ref_lf_cat.append(ref_lf.permute(0, 2, 3, 1).reshape(-1, C))
            ref_hf_cat.append(ref_hf.permute(0, 2, 3, 1).reshape(-1, C))
    
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
            features_raw = encoder(image)
            test_lf_list, test_hf_list = [], []
            
            for l in range(args.feature_levels):
                lf, hf = wav_filter.get_LF_HF(features_raw[l])
                test_lf_list.append(lf)
                test_hf_list.append(hf)
            
            rfeatures = get_freq_matched_residuals_infer(
                test_lf_list, test_hf_list, ref_lf_cat, ref_hf_cat, 
                alpha=args.blend_alpha, pos_flag=True, device=device
            )
            
            rfeatures = constraintor(*rfeatures)
        
            for l in range(args.feature_levels):
                e = rfeatures[l]  
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                
                # ★ 修正ポイント: estimatorの条件付けには一律で args.pos_embed_dim を使用する
                pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(device).unsqueeze(0).repeat(bs, 1, 1, 1)
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

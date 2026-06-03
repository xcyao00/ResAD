import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

from models.modules import get_position_encoding
from models.utils import get_logp
from utils import get_residual_features
from utils import calculate_metrics
from losses.utils import get_logp_a

warnings.filterwarnings('ignore')

def get_matched_ref_features(features, ref_features):
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        feature_n = F.normalize(feature, p=2, dim=1)
        coreset = ref_features[layer_id]
        coreset_n = F.normalize(coreset, p=2, dim=1)
        dist = feature_n @ coreset_n.T
        cidx = torch.argmax(dist, dim=1)
        index_feats = coreset[cidx]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)
    return matched_ref_features

def validate(args, encoder, constraintor, estimators, test_loader, ref_features, device, class_name):
    constraintor.eval()
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
            
            mfeatures = get_matched_ref_features(features, ref_features)
            
            rfeatures = get_residual_features(features, mfeatures, pos_flag=True)
            
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

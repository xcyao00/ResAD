import warnings
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F

from datasets import get_normal_image_paths
from models.modules import get_position_encoding
from models.utils import get_logp
from utils import get_residual_features, get_matched_ref_features
from utils import calculate_metrics, applying_EFDM
from losses.utils import get_logp_a

warnings.filterwarnings('ignore')


def validate(args, encoder, vq_ops, constraintor, estimators, test_loader, ref_features, device, class_name):
    vq_ops.eval()
    constraintor.eval()
    for estimator in estimators:  
        estimator.eval()

    # normal_image_paths = get_normal_image_paths('/path/to/your/dataset', class_name, dataset='btad')  # normal train images
    # matched_indices = np.load(f'./aligned/indices/{dataset_name}/{class_name}_knn_indices.npy')
    
    label_list, gt_mask_list = [], []
    logps1_list = [list() for _ in range(args.feature_levels)]
    logps2_list = [list() for _ in range(args.feature_levels)]
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating")
    for idx, batch in enumerate(test_loader):
        progress_bar.update(1)
        
        image, label, mask, _ = batch    
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(bool).ravel())

        # get ref_features from aligned ref images
        # indices = matched_indices[idx]
        # rimage_paths = [normal_image_paths[ind] for ind in indices]
        # rimages = load_and_transform_vision_data(rimage_paths, device)
        # with torch.no_grad():
        #     ref_features = encoder.encode_image_from_tensors(rimages.to(device))
        #     for l in range(len(ref_features)):
        #         _, _, c = ref_features[l].shape
        #         ref_features[l] = ref_features[l].reshape(-1, c)
        
        image = image.to(device)
        size = image.shape[-1]
        
        with torch.no_grad():
            if args.backbone == 'wide_resnet50_2':
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
            
            fdm_features = vq_ops(rfeatures, train=False)
            rfeatures = applying_EFDM(rfeatures, fdm_features, alpha=args.fdm_alpha)
            rfeatures = constraintor(*rfeatures)
        
            for l in range(args.feature_levels):
                e = rfeatures[l]  # BxCxHxW
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                
                # (bs, 128, h, w)
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
                
                logps_a = get_logp_a(dim, z, log_jac_det)  # logps corresponding to abnormal distribution
                logits = torch.stack([logps, logps_a], dim=-1)  # (N, 2)
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
        logps-= torch.max(logps) # normalize log-likelihoods to (-Inf:0] by subtracting a constant
        probs = torch.exp(logps) # convert to probs in range [0:1]
        # upsample
        normal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(normal_map[0])
    for l in range(feature_levels):
        scores += normal_map[l]

    # normality score to anomaly score
    scores = scores.max() - scores 
    
    #if class_name in ['pill', 'cable', 'capsule', 'screw']:
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores


def aggregate_anomaly_scores(logps_list, feature_levels=3, class_name=None, size=224):
    abnormal_map = [list() for _ in range(feature_levels)]
    for l in range(feature_levels):
        probs = torch.cat(logps_list[l], dim=0)  
        # upsample
        abnormal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(abnormal_map[0])
    for l in range(feature_levels):
        scores += abnormal_map[l]
    scores /= feature_levels
    
    for i in range(scores.shape[0]):
        scores[i] = gaussian_filter(scores[i], sigma=4)

    return scores

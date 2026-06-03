import os
import math
import random
from typing import List, Dict
from PIL import Image
import numpy as np
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as T
from datasets.mvtec import MVTEC
from datasets.visa import VISA


class BoundaryAverager:
    def __init__(self, num_levels=3):
        self.boundaries = [0 for _ in range(num_levels)]
    
    def update_boundary(self, boundary, level, momentum=0.9):
        lvl_boundary = self.boundaries[level]
        lvl_boundary = lvl_boundary * momentum + (1 - momentum) * boundary
        self.boundaries[level] = lvl_boundary
        
    def get_boundary(self, level):
        return self.boundaries[level]
    
    
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def get_matched_ref_features(features: List[Tensor], ref_features: List[Tensor]) -> List[Tensor]:
    """
    Get matched reference features for one class.
    """
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        feature_n = F.normalize(feature, p=2, dim=1)
        coreset = ref_features[layer_id]  # (N2, C)
        coreset_n = F.normalize(coreset, p=2, dim=1)
        dist = feature_n @ coreset_n.T
        cidx = torch.argmax(dist, dim=1)
        index_feats = coreset[cidx]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)
    
    return matched_ref_features
def get_matched_ref_features_top(features: List[Tensor], ref_features: List[Tensor], rank: int = 0) -> List[Tensor]:
    """
    Get matched reference features for one class.
    Args:
        rank (int): 取得する類似度の順位。0で最も似ているもの、1で2番目...を指定。
    """
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        feature_n = F.normalize(feature, p=2, dim=1)
        coreset = ref_features[layer_id]  # (N2, C)
        coreset_n = F.normalize(coreset, p=2, dim=1)
        dist = feature_n @ coreset_n.T

        # --- 変更箇所: rankに応じてインデックスを取得 ---
        if rank == 0:
            cidx = torch.argmax(dist, dim=1)
        else:
            # 上位 (rank + 1) 個を取得し、その最後の要素（指定された順位のもの）を取得
            # kがメモリバンクサイズを超えないように注意が必要ですが、通常は十分大きいためこのまま実装します
            _, topk_indices = torch.topk(dist, k=rank + 1, dim=1)
            cidx = topk_indices[:, -1]
        # ----------------------------------------------

        index_feats = coreset[cidx]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)
    
    return matched_ref_features


_SOFT_TOPK_DEBUG_PRINTED = set()


def _debug_soft_topk_match_shapes(prefix, layer_id, feature_shape, coreset_shape, matched_shape):
    key = (prefix, layer_id)
    if key in _SOFT_TOPK_DEBUG_PRINTED:
        return
    print(
        f"[SoftTopKMatching] {prefix} level {layer_id}: "
        f"feature={feature_shape}, coreset={coreset_shape}, matched={matched_shape}"
    )
    _SOFT_TOPK_DEBUG_PRINTED.add(key)


def _soft_topk_match_flat(feature, coreset, topk=5, tau=0.05, chunk_size=8192):
    if tau <= 0:
        raise ValueError("match_tau must be > 0 for soft_topk matching.")
    if topk <= 0:
        raise ValueError("match_topk must be > 0 for soft_topk matching.")
    if chunk_size <= 0:
        raise ValueError("match_chunk_size must be > 0 for soft_topk matching.")

    feature_n = F.normalize(feature, p=2, dim=1)
    coreset_n = F.normalize(coreset, p=2, dim=1)
    k = min(topk, coreset.shape[0])

    outputs = []
    for start in range(0, feature.shape[0], chunk_size):
        end = min(start + chunk_size, feature.shape[0])
        f_chunk = feature_n[start:end]
        sim = f_chunk @ coreset_n.T
        topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)
        weights = torch.softmax(topk_sim / tau, dim=1)
        topk_ref = coreset[topk_idx]
        matched = torch.sum(weights.unsqueeze(-1) * topk_ref, dim=1)
        outputs.append(matched)

    return torch.cat(outputs, dim=0)


def get_matched_ref_features_by_mode(
    features,
    ref_features,
    match_mode="hard",
    topk=5,
    tau=0.05,
    chunk_size=8192,
):
    if match_mode == "hard":
        return get_matched_ref_features(features, ref_features)
    if match_mode != "soft_topk":
        raise ValueError(f"Unsupported match_mode: {match_mode}")

    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        coreset = ref_features[layer_id]
        matched_flat = _soft_topk_match_flat(
            feature_flat,
            coreset,
            topk=topk,
            tau=tau,
            chunk_size=chunk_size,
        )
        matched = matched_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        _debug_soft_topk_match_shapes(
            "validate",
            layer_id,
            tuple(feature.shape),
            tuple(coreset.shape),
            tuple(matched.shape),
        )
        matched_ref_features.append(matched)

    return matched_ref_features


def get_residual_features(features: List[Tensor], ref_features: List[Tensor], pos_flag: bool = False) -> List[Tensor]:
    residual_features = []
    for layer_id in range(len(features)):
        fi = features[layer_id]  # (B, dim, h, w)
        pi = ref_features[layer_id]  # (B, dim, h, w)
        
        if not pos_flag:
            ri = fi - pi
        else:
            ri = F.mse_loss(fi, pi, reduction='none')
        residual_features.append(ri)
    
    return residual_features


def get_residual_features_by_mode(features: List[Tensor], ref_features: List[Tensor], mode: str = "sq") -> List[Tensor]:
    residual_features = []
    for fi, pi in zip(features, ref_features):
        if mode == "sq":
            ri = F.mse_loss(fi, pi, reduction='none')
        elif mode == "abs":
            ri = torch.abs(fi - pi)
        elif mode == "signed":
            ri = fi - pi
        else:
            raise ValueError(f"Unsupported residual_mode: {mode}")
        residual_features.append(ri)

    return residual_features
import torch

def get_image_level_matched_features(features, ref_features):
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        
        coreset = ref_features[layer_id]
        K = coreset.shape[0] // (H * W)
        coreset_spatial = coreset.view(K, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        feat_flat = feature.reshape(B, 1, -1)
        core_flat = coreset_spatial.reshape(1, K, -1)
        
        feat_norm = F.normalize(feat_flat, p=2, dim=2)
        core_norm = F.normalize(core_flat, p=2, dim=2)
        sim = torch.sum(feat_norm * core_norm, dim=2)
        
        best_idx = torch.argmax(sim, dim=1)
        matched = coreset_spatial[best_idx]
        matched_ref_features.append(matched)
        
    return matched_ref_features

def get_mc_image_level_matched_features(features, class_names, ref_features):
    matched_ref_features = [[] for _ in range(len(features))]
    for idx, c in enumerate(class_names):
        ref_features_c = ref_features[c]
        
        for layer_id in range(len(features)):
            feature = features[layer_id][idx:idx+1]
            _, C, H, W = feature.shape
            
            coreset = ref_features_c[layer_id]
            K = coreset.shape[0] // (H * W)
            coreset_spatial = coreset.view(K, H, W, C).permute(0, 3, 1, 2).contiguous()
            
            feat_flat = feature.reshape(1, 1, -1)
            core_flat = coreset_spatial.reshape(1, K, -1)
            
            feat_norm = F.normalize(feat_flat, p=2, dim=2)
            core_norm = F.normalize(core_flat, p=2, dim=2)
            sim = torch.sum(feat_norm * core_norm, dim=2)
            
            best_idx = torch.argmax(sim, dim=1)
            matched = coreset_spatial[best_idx].squeeze(0)
            matched_ref_features[layer_id].append(matched)
            
    matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    return matched_ref_features
def get_fourier_residual_features(features, mfeatures, pos_flag=True):
    """
    周波数領域で残差を計算する関数
    features: テスト画像の特徴量リスト [B, C, H, W]
    mfeatures: マッチングされた参照画像の特徴量リスト [B, C, H, W]
    """
    rfeatures = []
    for i in range(len(features)):
        f, mf = features[i], mfeatures[i]
        
        # 1. 2D FFTで空間から周波数領域へ変換
        f_fft = torch.fft.fft2(f, norm="ortho")
        mf_fft = torch.fft.fft2(mf, norm="ortho")
        
        # 2. 振幅 (Amplitude) と 位相 (Phase) に分離
        f_amp, f_pha = torch.abs(f_fft), torch.angle(f_fft)
        mf_amp, mf_pha = torch.abs(mf_fft), torch.angle(mf_fft)
        
        # 3. 振幅の残差を計算 (ここがキズに反応する)
        # 位相の残差は空間構造の歪みを表すが、位置ズレに寛容にするためテスト画像の位相を保持する
        res_amp = torch.abs(f_amp - mf_amp)
        
        # 4. 残差振幅とテスト画像の位相を結合して複素数に戻す
        res_fft_new = res_amp * torch.exp(1j * f_pha)
        
        # 5. 逆フーリエ変換 (IFFT) で空間領域の残差マップに戻す
        res_f = torch.fft.ifft2(res_fft_new, norm="ortho").real
        #元のMSEに合わせるために2乗する
        if pos_flag:
            res_f = torch.pow(res_f, 2) # または torch.abs(res_f)      
            
        rfeatures.append(res_f)
        
    return rfeatures        

def load_reference_features(root_dir: str, class_name: str, device: torch.device) -> List[Tensor]:
    """
    Load reference features for one class.
    """
    layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
    layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
    layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
    
    layer1_refs = torch.from_numpy(layer1_refs).to(device)
    layer2_refs = torch.from_numpy(layer2_refs).to(device)
    layer3_refs = torch.from_numpy(layer3_refs).to(device)
    
    return layer1_refs, layer2_refs, layer3_refs


def get_random_normal_images(root, class_name, num_shot=4):
    if class_name in MVTEC.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'train', 'good')
    elif class_name in VISA.CLASS_NAMES:
        root_dir = os.path.join(root, class_name, 'Data', 'Images', 'Normal')
    else:
        raise ValueError('Unrecognized class_name!')
    filenames = os.listdir(root_dir)
    n_idxs = np.random.randint(len(filenames), size=num_shot)
    n_idxs = n_idxs.tolist()
    normal_paths = []
    for n_idx in n_idxs:
        normal_paths.append(os.path.join(root_dir, filenames[n_idx]))
    
    return normal_paths


def get_mc_reference_features(encoder, root, class_names, device, num_shot=4, img_size=224):
    """
    Get reference features for multiple classes.
    """
    reference_features = {}
    class_names = np.unique(class_names)
    for class_name in class_names:
        normal_paths = get_random_normal_images(root, class_name, num_shot)
        images = load_and_transform_vision_data(normal_paths, device, image_size=img_size)
        with torch.no_grad():
            features = encoder(images)
            for l in range(len(features)):
                bs, c, h, w = features[l].shape
                features[l] = features[l].permute(0, 2, 3, 1).reshape(-1, c)
            reference_features[class_name] = features
    return reference_features
def get_mc_reference_features_wav(encoder, root, class_names, device, num_shot=4, wav_filter=None, img_size=224):
    """
    Get reference features for multiple classes.
    """
    reference_features = {}
    class_names = np.unique(class_names)
    for class_name in class_names:
        normal_paths = get_random_normal_images(root, class_name, num_shot)
        images = load_and_transform_vision_data(normal_paths, device, image_size=img_size)
        with torch.no_grad():
            features = encoder(images)
            
            # ======== 【追加】 ========
            # 平坦化(reshape)される前に空間情報(H, W)を保ったままウェーブレット変換を適用
            if wav_filter is not None:
                features = [wav_filter(f) for f in features]
            # ========================
            
            for l in range(len(features)):
                bs, c, h, w = features[l].shape
                features[l] = features[l].permute(0, 2, 3, 1).reshape(-1, c)
            reference_features[class_name] = features
    return reference_features

def load_and_transform_vision_data(image_paths, device, image_size=224):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = T.Compose([
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])])
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)


def get_mc_matched_ref_features(features: List[Tensor], class_names: List[str],
                                ref_features: Dict[str, List[Tensor]]) -> List[Tensor]:
    """
    Get matched reference features for multiple classes.
    """
    matched_ref_features = [[] for _ in range(len(features))]
    for idx, c in enumerate(class_names):  # for each image
        ref_features_c = ref_features[c]
        
        for layer_id in range(len(features)):  # for all layers of one image
            feature = features[layer_id][idx:idx+1]
            _, C, H, W = feature.shape
            
            feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
            feature_n = F.normalize(feature, p=2, dim=1)
            coreset = ref_features_c[layer_id]  # (N2, C)
            coreset_n = F.normalize(coreset, p=2, dim=1)
            dist = feature_n @ coreset_n.T  # (N1, N2)
            cidx = torch.argmax(dist, dim=1)
            index_feats = coreset[cidx]
            index_feats = index_feats.permute(1, 0).reshape(C, H, W)
            matched_ref_features[layer_id].append(index_feats)
            
    matched_ref_features = [torch.stack(item, dim=0) for item in matched_ref_features]
    
    return matched_ref_features


def get_mc_matched_ref_features_by_mode(
    features,
    class_names,
    ref_features,
    match_mode="hard",
    topk=5,
    tau=0.05,
    chunk_size=8192,
):
    if match_mode == "hard":
        return get_mc_matched_ref_features(features, class_names, ref_features)
    if match_mode != "soft_topk":
        raise ValueError(f"Unsupported match_mode: {match_mode}")

    matched_ref_features = [[] for _ in range(len(features))]
    coreset_shapes = [None for _ in range(len(features))]
    for idx, c in enumerate(class_names):
        ref_features_c = ref_features[c]

        for layer_id in range(len(features)):
            feature = features[layer_id][idx:idx + 1]
            _, C, H, W = feature.shape
            feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
            coreset = ref_features_c[layer_id]
            if coreset_shapes[layer_id] is None:
                coreset_shapes[layer_id] = tuple(coreset.shape)
            matched_flat = _soft_topk_match_flat(
                feature_flat,
                coreset,
                topk=topk,
                tau=tau,
                chunk_size=chunk_size,
            )
            matched = matched_flat.permute(1, 0).reshape(C, H, W)
            matched_ref_features[layer_id].append(matched)

    stacked_features = []
    for layer_id, item in enumerate(matched_ref_features):
        matched = torch.stack(item, dim=0)
        _debug_soft_topk_match_shapes(
            "train",
            layer_id,
            tuple(features[layer_id].shape),
            coreset_shapes[layer_id],
            tuple(matched.shape),
        )
        stacked_features.append(matched)
    return stacked_features


def calculate_metrics(scores, labels, gt_masks, pro=True, only_max_value=True):
    """
    Args:
        scores (np.ndarray): shape (N, H, W).
        labels (np.ndarray): shape (N, ), 0 for normal, 1 for abnormal.
        gt_masks (np.ndarray): shape (N, H, W).
    """
    #pixel AUC error analysis
    from scipy.stats import rankdata
    if False:        
        scores_flat = np.maximum(scores.reshape(-1,1),0)
        ranks_flat = rankdata(scores_flat)
        scores = scores_flat.reshape(N,H,W)
        ranks = ranks_flat.reshape(N,H,W)
        mask_rank = (gt_masks*ranks).sum(dim=(1,2))
        unmask_rank = ((1-gt_masks)*ranks).sum(dim=(1,2))
        fn_mask=(rankdata(rankdata(mask_rank))[labels==1])[-10:]
        fp_unmask = (rankdata(rankdata(unmask_rank))[labels==0])[:10]
        
        score_max = scores.max(dim=(1,2))
        false_negatives=(rankdata(rankdata(score_max))[labels==1])[-10:]
        false_positives=(rankdata(rankdata(score_max))[labels==0])[:10]

    # average precision
    pix_ap = round(average_precision_score(gt_masks.flatten(), scores.flatten()), 5)
    # f1 score, f1 score is to balance the precision and recall
    # f1 score is high means the precision and recall are both high
    precisions, recalls, _ = precision_recall_curve(gt_masks.flatten(), scores.flatten())
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    pix_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
    # roc auc
    pix_auc = round(roc_auc_score(gt_masks.flatten(), scores.flatten()), 5)
    
    _, h, w = scores.shape
    size = h * w
    if only_max_value:
        topks = [1]
    else:
        topks = [int(size*p) for p in np.arange(0.01, 0.41, 0.01)]
        topks = [1, 100] + topks
    img_aps, img_aucs, img_f1_scores = [], [], []
    for topk in topks:
        img_scores = get_image_scores(scores, topk)
        img_ap = round(average_precision_score(labels, img_scores), 5)
        precisions, recalls, _ = precision_recall_curve(labels, img_scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        img_f1_score = round(np.max(f1_scores[np.isfinite(f1_scores)]), 5)
        img_auc = round(roc_auc_score(labels, img_scores), 5)
        img_aps.append(img_ap)
        img_aucs.append(img_auc)
        img_f1_scores.append(img_f1_score)
    img_ap, img_auc, img_f1_score = np.max(img_aps), np.max(img_aucs), np.max(img_f1_scores)  
    pix_aupro = calculate_aupro(gt_masks, scores)
    
    return img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro


def get_image_scores(scores, topk=1):
    scores_ = torch.from_numpy(scores)
    img_scores = torch.topk(scores_.reshape(scores_.shape[0], -1), topk, dim=1)[0]
    img_scores = torch.mean(img_scores, dim=1)
    img_scores = img_scores.cpu().numpy()
        
    return img_scores


def calculate_aupro(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    if fprs.shape[0] <= 2:
        return 0.5
    else:
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        pro_auc = auc(fprs, pros[idxes])
        return pro_auc


def applying_EFDM(input_features_list, ref_features_list, alpha=0.5):
    """
    Args:
        input_features (Tensor): shape of (B, C, H, W).
        ref_features (Tensor): normal reference features, (B, C, H, W).
    """
    alpha = 1 - alpha
    aligned_features_list = []
    for l in range(len(input_features_list)):
        input_features, ref_features = input_features_list[l], ref_features_list[l]
        B, C, W, H = input_features.shape

        input_features_r = input_features.reshape(B, C, -1)
        ref_features_r = ref_features.reshape(B, C, -1)

        sorted_input_features, inds = torch.sort(input_features_r)
        sorted_ref_features, _ = torch.sort(ref_features_r)
        aligned_features = sorted_input_features + (sorted_ref_features - sorted_input_features) * alpha
        inv_inds = inds.argsort(-1)
        aligned_features = aligned_features.gather(-1, inv_inds)
        aligned_features = aligned_features.view(B, C, W, H)
        aligned_features_list.append(aligned_features)

    return aligned_features_list
    '''
def load_weights(encoder, decoders, filename):#12/16追加
    #path = os.path.join(WEIGHT_DIR, filename)
    state = torch.load(filename)
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])] #12/16変更
    print('Loading weights from {}'.format(filename))
    '''
def load_weights(encoder, decoders, filename):
    state = torch.load(filename, map_location='cpu')
    print(f'Loading weights from {filename}')

    # --- Encoder のロード ---
    if 'encoder_state_dict' in state:
        enc_msg = encoder.load_state_dict(state['encoder_state_dict'], strict=False)
        all_keys = len(encoder.state_dict().keys())
        loaded_keys = all_keys - len(enc_msg.missing_keys)
        print(f"--- Encoder Load Report ---")
        print(f"  Successfully loaded: {loaded_keys}/{all_keys}")
    else:
        print("  [ERROR] 'encoder_state_dict' not found.")

    # --- Decoders のロード ---
    if 'decoder_state_dict' in state:
        print(f"--- Decoders Detailed Report ---")
        for i, (decoder, d_state) in enumerate(zip(decoders, state['decoder_state_dict'])):
            dec_msg = decoder.load_state_dict(d_state, strict=False)
            
            all_keys = set(decoder.state_dict().keys())
            missing_keys = set(dec_msg.missing_keys)
            loaded_keys_count = len(all_keys) - len(missing_keys)
            
            print(f"  Decoder {i}: Loaded {loaded_keys_count}/{len(all_keys)} parameters.")
            # 修正箇所: インデントを削除しました
            for key in sorted(list(all_keys)):
                print(f"      - {key}")          

            if len(missing_keys) > 0:
                print(f"    [Missing parameters in Decoder {i}]")
                # 読み込めなかった変数名をすべて書き出す
                for key in sorted(list(missing_keys)):
                    print(f"      - {key}")
            else:
                print(f"    All parameters loaded successfully for Decoder {i}.")
    else:
        print("  [ERROR] 'decoder_state_dict' not found.")
        
def load_weights_ada(adapter, filename):
    #path = os.path.join(WEIGHT_DIR, filename)
    state = torch.load(filename)
    #encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    #decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])]
    adapters = [adapters.load_state_dict(state, strict=False) for adapter, state in zip(adapters, state['adapter_state_dict'])]#modified 1/8
    print('Loading weights from {}'.format(filename))
def get_soft_matched_features(features: List[Tensor], ref_features: List[Tensor], tau=0.05) -> List[Tensor]:
    """
    評価用: Soft Attentionを用いた滑らかな特徴マッチング
    tau (Temperature): 値が小さいほどArgmaxに近づき（鮮明）、大きいほど平均に近づく（滑らか）
    """
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape

        # [B*H*W, C] に変形
        feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        feature_n = F.normalize(feature_flat, p=2, dim=1)

        coreset = ref_features[layer_id]  # [K*H*W, C] (Kはショット数)
        coreset_n = F.normalize(coreset, p=2, dim=1)

        # 1. すべての参照ピクセルとの類似度を計算
        sim = feature_n @ coreset_n.T  # [B*H*W, K*H*W]

        # 2. Temperature付きSoftmaxで重み（Attention）を計算
        attn = F.softmax(sim / tau, dim=1)  # [B*H*W, K*H*W]

        # 3. 参照ピクセルを重み付きでブレンド
        index_feats = attn @ coreset  # [B*H*W, C]

        # 4. 元の画像形状に戻す
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        matched_ref_features.append(index_feats)
    
    return matched_ref_features

def get_mc_soft_matched_features(features: List[Tensor], class_names: List[str], ref_features: Dict[str, List[Tensor]], tau=0.05) -> List[Tensor]:
    """
    学習用: マルチクラス対応のSoft Attentionマッチング
    """
    matched_ref_features = [[] for _ in range(len(features))]
    for idx, c in enumerate(class_names):  
        ref_features_c = ref_features[c]
        
        for layer_id in range(len(features)):  
            feature = features[layer_id][idx:idx+1]
            _, C, H, W = feature.shape
            
            feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
            feature_n = F.normalize(feature_flat, p=2, dim=1)
            
            coreset = ref_features_c[layer_id]
            coreset_n = F.normalize(coreset, p=2, dim=1)
            
            sim = feature_n @ coreset_n.T 
            attn = F.softmax(sim / tau, dim=1) 
            
            index_feats = attn @ coreset 
            index_feats = index_feats.reshape(1, H, W, C).permute(0, 3, 1, 2).contiguous()
            matched_ref_features[layer_id].append(index_feats)
            
    matched_ref_features = [torch.cat(item, dim=0) for item in matched_ref_features]
    return matched_ref_features
def compute_osp_matrices_from_refs(ref_features_tuple, keep_variance=0.95):
    """
    参照特徴量 (正常データ) から、レイヤーごとの射影行列と平均ベクトルを計算する。
    ref_features_tuple: (layer1_refs, layer2_refs, layer3_refs) などのタプル
                        各 tensor は [N, C] または [N, C, H, W] などの形状
    """
    proj_matrices = []
    means = []
    
    for ref_feat in ref_features_tuple:
        # 形状を [Batch*H*W, Channels] の2次元に平坦化する
        if ref_feat.dim() == 4:
            B, C, H, W = ref_feat.shape
            flat_ref = ref_feat.permute(0, 2, 3, 1).reshape(-1, C)
        elif ref_feat.dim() == 3:
            B, L, C = ref_feat.shape
            flat_ref = ref_feat.reshape(-1, C)
        else:
            flat_ref = ref_feat
            C = flat_ref.shape[-1]
            
        mean = flat_ref.mean(dim=0, keepdim=True)
        centered = flat_ref - mean
        
        # 特異値分解 (SVD) を計算
        U, S, V = torch.linalg.svd(centered.cpu(), full_matrices=False)
        
        # 寄与率から上位 k 次元を決定
        var = (S ** 2) / (centered.size(0) - 1)
        cum_var = torch.cumsum(var, dim=0) / var.sum()
        k = torch.searchsorted(cum_var, keep_variance).item() + 1
        
        basis = V[:k, :].T.to(ref_feat.device)  # [C, k]
        proj_matrix = torch.mm(basis, basis.T)  # [C, C]
        
        proj_matrices.append(proj_matrix)
        means.append(mean.to(ref_feat.device))
        
    return proj_matrices, means

def apply_osp(rfeatures_list, proj_matrices, means):
    """
    抽出された残差リストに対して、正常空間成分を削り落とす。
    """
    osp_residuals = []
    for i, rfeat in enumerate(rfeatures_list):
        B, C, H, W = rfeat.shape
        r_flat = rfeat.permute(0, 2, 3, 1).reshape(-1, C)
        
        # 平行成分（正常なズレ）を計算
        r_parallel = torch.mm(r_flat - means[i], proj_matrices[i])
        
        # 直交成分（純粋な異常）を残す
        r_orthogonal = (r_flat - means[i]) - r_parallel
        
        # 元の形状に戻す
        r_orthogonal = r_orthogonal.reshape(B, H, W, C).permute(0, 3, 1, 2)
        osp_residuals.append(r_orthogonal)
        
    return osp_residuals
import torch
import torch.nn.functional as F
import torch.nn as nn
class HaarWaveletFilter(nn.Module):
    def __init__(self, low_freq_weight=0.1, high_freq_weight=1.2):
        super().__init__()
        self.lf_w = low_freq_weight
        self.hf_w = high_freq_weight
        
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        
        self.register_buffer('k_ll', ll.view(1, 1, 2, 2))
        self.register_buffer('k_hl', hl.view(1, 1, 2, 2))
        self.register_buffer('k_lh', lh.view(1, 1, 2, 2))
        self.register_buffer('k_hh', hh.view(1, 1, 2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        ll = F.conv2d(x, self.k_ll.expand(C, 1, 2, 2), stride=2, groups=C)
        hl = F.conv2d(x, self.k_hl.expand(C, 1, 2, 2), stride=2, groups=C)
        lh = F.conv2d(x, self.k_lh.expand(C, 1, 2, 2), stride=2, groups=C)
        hh = F.conv2d(x, self.k_hh.expand(C, 1, 2, 2), stride=2, groups=C)
        
        ll = ll * self.lf_w
        hl = hl * self.hf_w
        lh = lh * self.hf_w
        hh = hh * self.hf_w
        
        out = F.conv_transpose2d(ll, self.k_ll.expand(C, 1, 2, 2), stride=2, groups=C) + \
              F.conv_transpose2d(hl, self.k_hl.expand(C, 1, 2, 2), stride=2, groups=C) + \
              F.conv_transpose2d(lh, self.k_lh.expand(C, 1, 2, 2), stride=2, groups=C) + \
              F.conv_transpose2d(hh, self.k_hh.expand(C, 1, 2, 2), stride=2, groups=C)
        return out

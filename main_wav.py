import os
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import torch
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import train
from validate_wav import validate
from datasets.mvtec import MVTEC, MVTECANO
from datasets.visa import VISA, VISAANO
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS
from datasets.capsules import CAPSULES, CAPSULESANO

from models.fc_flow import load_flow_model
from models.modules import MultiScaleConv
from models.dinov2_backbone import DINOv2BackboneWrapper, DINOV2_BACKBONES, DINOV2_FEATURE_MODES
from models.dinov2_backbone import dinov2_shape_test, print_dinov2_config
from models.soft_codebook import SoftCodebookAdapterList
from models.vq import MultiScaleVQ
from utils import init_seeds, get_residual_features_by_mode, get_mc_matched_ref_features_by_mode
from utils import get_mc_reference_features, get_mc_reference_features_wav
from utils import BoundaryAverager
from raw_vqops import apply_raw_vqops_if_enabled, print_raw_vqops_config, train_raw_vqops_if_enabled
from residual_wavelet import apply_feature_wavelet_filter, apply_residual_wavelet_filter, residual_wavelet_shape_test
from losses.loss import calculate_log_barrier_bi_occ_loss
from classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS
from classes import MVTEC_TO_MVTEC, VISA_TO_VISA
from classes import CAPSULES_TO_CAPSULES

warnings.filterwarnings('ignore')

TOTAL_SHOT = 4  
FIRST_STAGE_EPOCH = 10
SETTINGS = {'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
            'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
            'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
            'mvtec_to_brats': MVTEC_TO_BRATS,'mvtec_to_mvtec':MVTEC_TO_MVTEC, 'visa_to_visa':VISA_TO_VISA, 'capsules_to_capsules': CAPSULES_TO_CAPSULES}


def print_soft_codebook_config(args):
    print("[SoftCodebook] use_soft_codebook:", args.use_soft_codebook)
    print("[SoftCodebook] soft_cb_pos:", args.soft_cb_pos)
    print("[SoftCodebook] soft_cb_k:", args.soft_cb_k)
    print("[SoftCodebook] soft_cb_tau:", args.soft_cb_tau)
    print("[SoftCodebook] soft_cb_gamma:", args.soft_cb_gamma)
    print("[SoftCodebook] soft_cb_warmup_epochs:", args.soft_cb_warmup_epochs)
    print("[SoftCodebook] soft_cb_conf_gate:", args.soft_cb_conf_gate)
    print("[SoftCodebook] soft_cb_gate_threshold:", args.soft_cb_gate_threshold)
    print("[SoftCodebook] soft_cb_gate_temp:", args.soft_cb_gate_temp)
    if args.use_soft_codebook and args.soft_cb_warmup_epochs > 0:
        print("[SoftCodebook] gamma warmup: epoch 0 starts with gamma_eff=0.0")


def print_wavelet_config(args):
    print("[Wavelet] wav_on:", args.wav_on)
    print("[Wavelet] wav_mode:", args.wav_mode)
    print("[Wavelet] feature_wav_mode:", args.feature_wav_mode)
    print("[Wavelet] hf_skip_alpha:", args.hf_skip_alpha)
    print("[Wavelet] wav_hf_normalize:", args.wav_hf_normalize)


def apply_feature_wavelet_from_args(args, features):
    return apply_feature_wavelet_filter(
        features,
        wave=args.wave,
        feature_wav_mode=args.feature_wav_mode,
        hf_weight=args.hf_weight,
        ll_skip_alpha=args.ll_skip_alpha,
        hf_skip_alpha=args.hf_skip_alpha,
        wav_hf_normalize=args.wav_hf_normalize,
    )


def assert_feature_shapes_match(features, matched_features, prefix):
    for level, (feature, matched) in enumerate(zip(features, matched_features)):
        if feature.shape != matched.shape:
            raise ValueError(
                f"{prefix} level {level} shape mismatch: "
                f"feature={tuple(feature.shape)}, matched={tuple(matched.shape)}"
            )


def main(args):
    if args.use_raw_vqops and args.use_soft_codebook:
        raise ValueError("Do not use raw VQOps and SoftCB together in this ablation.")
    if args.setting in SETTINGS.keys():
        CLASSES = SETTINGS[args.setting]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")

    # TODO: Consider adding a DINOv2-specific normalization option and compare it with the existing w50 normalization.
                
    if args.classes == 'capsules': 
        train_dataset1 = CAPSULES(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                               normalize="w50",
                               img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(
            train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        train_dataset2 = CAPSULESANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                                  normalize='w50',
                                  img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(
            train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )    
    elif CLASSES['seen'][0] in MVTEC.CLASS_NAMES: 
        train_dataset1 = MVTEC(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                               normalize="w50",
                               img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(
            train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        train_dataset2 = MVTECANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                                  normalize='w50',
                                  img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(
            train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
    else: 
        train_dataset1 = VISA(args.train_dataset_dir, class_name=CLASSES['seen'], train=True,
                               normalize="w50",
                               img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(
            train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        train_dataset2 = VISAANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                                 normalize="w50",
                                 img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(
            train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        
    if args.backbone == 'wide_resnet50_2':
        encoder = timm.create_model('wide_resnet50_2', features_only=True,
                out_indices=(1, 2, 3), pretrained=True).eval() 
        encoder = encoder.to(args.device)
        feat_dims = encoder.feature_info.channels()
    elif args.backbone == 'tf_efficientnet_b6':
        encoder = timm.create_model('tf_efficientnet_b6', features_only=True,
                out_indices=(1, 2, 3), pretrained=True).eval() 
        encoder = encoder.to(args.device)
        feat_dims = encoder.feature_info.channels()
    elif args.backbone in DINOV2_BACKBONES:
        encoder = DINOv2BackboneWrapper(
            model_name=args.backbone,
            out_dims=(40, 72, 200),
            out_sizes=(56, 28, 14),
            freeze=True,
            feature_mode=args.dinov2_feature_mode,
            layers=args.dinov2_layers,
            proj_dim=args.dinov2_proj_dim,
        ).to(args.device)
        encoder.eval()
        feat_dims = encoder.feature_info.channels()
        print_dinov2_config(encoder, image_size=224)
        
    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)
    print("[Residual] residual_mode:", args.residual_mode)
    print("[Matching] match_mode:", args.match_mode)
    print("[Matching] match_topk:", args.match_topk)
    print("[Matching] match_tau:", args.match_tau)
    print("[Matching] match_chunk_size:", args.match_chunk_size)
    print_wavelet_config(args)
    print_raw_vqops_config(args)
    print_soft_codebook_config(args)

    raw_vq_ops = None
    optimizer_vq = None
    scheduler_vq = None
    if args.use_raw_vqops:
        raw_vq_ops = MultiScaleVQ(num_embeddings=args.num_embeddings, channels=feat_dims).to(args.device)
        optimizer_vq = torch.optim.Adam(raw_vq_ops.parameters(), lr=args.lr, weight_decay=0.0005)
        scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(optimizer_vq, milestones=[70, 90], gamma=0.1)
    
    # constraintorの初期化 (元のmain.pyに準拠)
    constraintor = MultiScaleConv(feat_dims).to(args.device)
    optimizer0 = torch.optim.Adam(constraintor.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, milestones=[70, 90], gamma=0.1)

    soft_codebook = None
    if args.use_soft_codebook:
        soft_codebook = SoftCodebookAdapterList(
            feat_dims,
            num_embeddings=args.soft_cb_k,
            tau=args.soft_cb_tau,
            gamma=args.soft_cb_gamma,
            warmup_epochs=args.soft_cb_warmup_epochs,
            conf_gate=args.soft_cb_conf_gate,
            gate_threshold=args.soft_cb_gate_threshold,
            gate_temp=args.soft_cb_gate_temp,
        ).to(args.device)
    
    # NFの初期化
    estimators = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    estimators = [decoder.to(args.device) for decoder in estimators]
    params = list(estimators[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(estimators[l].parameters())
    if soft_codebook is not None:
        params += list(soft_codebook.parameters())
    optimizer1 = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0005)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[70, 90], gamma=0.1)
    
    best_img_auc = 0
    N_batch = 8192
    
    for epoch in range(args.epochs):
        if raw_vq_ops is not None:
            raw_vq_ops.train()
        constraintor.train()
        if soft_codebook is not None:
            soft_codebook.train()
        for estimator in estimators:
            estimator.train()
            
        if epoch < FIRST_STAGE_EPOCH:
            train_loader = train_loader1
        else:
            train_loader = train_loader2
            
        train_loss_total, total_num = 0, 0
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
        
        for step, batch in enumerate(train_loader):
            progress_bar.update(1)
            images, _, masks, class_names = batch
            
            images = images.to(args.device)
            masks = masks.to(args.device)
            
            with torch.no_grad():
                features = encoder(images)
                if args.use_wav and args.wav_on == 'feature':
                    features = apply_feature_wavelet_from_args(args, features)
            
            if args.use_wav and args.wav_on == 'feature':
                ref_features = get_mc_reference_features_wav(
                    encoder,
                    args.train_dataset_dir,
                    class_names,
                    images.device,
                    args.train_ref_shot,
                    wav_filter=lambda feature: apply_feature_wavelet_from_args(args, [feature])[0],
                )
            else:
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
            assert_feature_shapes_match(features, mfeatures, "train matching")
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
                    hf_skip_alpha=args.hf_skip_alpha,
                    wav_hf_normalize=args.wav_hf_normalize,
                )
            
            lvl_masks = []
            for l in range(args.feature_levels):
                _, _, h, w = rfeatures[l].size()
                m = F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1)
                lvl_masks.append(m)

            if args.use_raw_vqops and args.raw_vq_pos == "pre_constraintor":
                loss_vq = train_raw_vqops_if_enabled(args, raw_vq_ops, optimizer_vq, rfeatures, lvl_masks)
                if loss_vq is not None:
                    train_loss_total += loss_vq.item()
                    total_num += 1
                rfeatures = apply_raw_vqops_if_enabled(
                    args,
                    raw_vq_ops,
                    rfeatures,
                    prefix="train_pre_constraintor",
                    loss_vq=loss_vq,
                )

            rfeatures_t = [rfeature.detach().clone() for rfeature in rfeatures]
            
            # constraintor 適用
            rfeatures = constraintor(*rfeatures)
            loss = 0
            for l in range(args.feature_levels):  
                e = rfeatures[l]  
                t = rfeatures_t[l]
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                t = t.permute(0, 2, 3, 1).reshape(-1, dim)
                m = lvl_masks[l]
                m = m.reshape(-1)
                
                loss_i, _, _ = calculate_log_barrier_bi_occ_loss(e, m, t)
                loss += loss_i
            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step()
            
            train_loss_total += loss.item()
            total_num += 1
            
            nf_features = [rfeature.detach().clone() for rfeature in rfeatures]
            if args.use_raw_vqops and args.raw_vq_pos == "post_constraintor":
                loss_vq = train_raw_vqops_if_enabled(args, raw_vq_ops, optimizer_vq, nf_features, lvl_masks)
                if loss_vq is not None:
                    train_loss_total += loss_vq.item()
                    total_num += 1
                nf_features = apply_raw_vqops_if_enabled(
                    args,
                    raw_vq_ops,
                    nf_features,
                    prefix="train_post_constraintor",
                    loss_vq=loss_vq,
                )
            loss, num = train(
                args,
                nf_features,
                estimators,
                optimizer1,
                masks,
                boundary_ops,
                epoch,
                N_batch=N_batch,
                FIRST_STAGE_EPOCH=FIRST_STAGE_EPOCH,
                soft_codebook=soft_codebook,
            )
            train_loss_total += loss
            total_num += num
        
        if scheduler_vq is not None:
            scheduler_vq.step()
        scheduler0.step()
        scheduler1.step()
               
        progress_bar.close()
        print(f"Epoch[{epoch}/{args.epochs}]: train_loss: {train_loss_total / total_num}")
        
        # --- 評価フェーズ ---
        if (epoch + 1) % args.eval_freq == 0:
            s1_res, s2_res, s_res = [], [], []
            test_ref_features = load_mc_reference_features(args.test_ref_feature_dir, CLASSES['unseen'], args.device, args.num_ref_shot)
            
            for class_name in CLASSES['unseen']:
                if args.classes == 'capsules':
                    test_dataset = CAPSULES(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)                            
                elif class_name in MVTEC.CLASS_NAMES:
                    test_dataset = MVTEC(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in VISA.CLASS_NAMES:
                    test_dataset = VISA(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in BTAD.CLASS_NAMES:
                    test_dataset = BTAD(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in MVTEC3D.CLASS_NAMES:
                    test_dataset = MVTEC3D(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in MPDD.CLASS_NAMES:
                    test_dataset = MPDD(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in MVTECLOCO.CLASS_NAMES:
                    test_dataset = MVTECLOCO(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                elif class_name in BRATS.CLASS_NAMES:
                    test_dataset = BRATS(args.test_dataset_dir, class_name=class_name, train=False, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                else:
                    raise ValueError('Unrecognized class name: {}'.format(class_name))
                
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
                
                metrics = validate(args, encoder, constraintor, soft_codebook, raw_vq_ops, estimators, test_loader, test_ref_features[class_name], args.device, class_name, epoch=epoch)
                
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics['scores']
                
                # 元の main.py と同じ詳細な print 文
                print("Epoch: {}, Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                    epoch, class_name, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
                s1_res.append(metrics['scores1'])
                s2_res.append(metrics['scores2'])
                s_res.append(metrics['scores'])
            
            s1_res = np.array(s1_res)
            s2_res = np.array(s2_res)
            s_res = np.array(s_res)
            img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1 = np.mean(s1_res, axis=0)
            img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2 = np.mean(s2_res, axis=0)
            img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(s_res, axis=0)
            
            # 元の main.py と同じ平均値の print 文
            print('(Logps) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1))
            print('(BScores) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2))
            print('(Merged) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
            
            if img_auc > best_img_auc:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                best_img_auc = img_auc
                state_dict = {'constraintor': constraintor.state_dict(),
                              'estimators': [estimator.state_dict() for estimator in estimators]}
                if soft_codebook is not None:
                    state_dict['soft_codebook'] = soft_codebook.state_dict()
                if raw_vq_ops is not None:
                    state_dict['raw_vq_ops'] = raw_vq_ops.state_dict()
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'{args.setting}_epoch_{epoch}_checkpoints.pth'))


def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, 'layer1.npy'))).to(device)
        layer2_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, 'layer2.npy'))).to(device)
        layer3_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, 'layer3.npy'))).to(device)
        K1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        K2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        K3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot
        refs[class_name] = (layer1_refs[:K1, :], layer2_refs[:K2, :], layer3_refs[:K3, :])
    return refs
                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="visa_to_mvtec")
    parser.add_argument('--classes', type=str, default="none")
    parser.add_argument('--train_dataset_dir', type=str, default="")
    parser.add_argument('--test_dataset_dir', type=str, default="")
    parser.add_argument('--test_ref_feature_dir', type=str, default="./ref_features/w50/mvtec_4shot")
    parser.add_argument('--bgadweight_dir', type=str, default="none")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")
    parser.add_argument('--rank', type=int, default="0")    
    
    # flow parameters
    parser.add_argument('--flow_arch', type=str, default='conditional_flow_model')
    parser.add_argument('--feature_levels', default=3, type=int)
    parser.add_argument('--coupling_layers', type=int, default=10)
    parser.add_argument('--clamp_alpha', type=float, default=1.9)
    parser.add_argument('--pos_embed_dim', type=int, default=256)
    parser.add_argument('--pos_beta', type=float, default=0.05)
    parser.add_argument('--margin_tau', type=float, default=0.1)
    parser.add_argument('--bgspp_lambda', type=float, default=1)
    
    parser.add_argument('--fdm_alpha', type=float, default=0.4) 
    parser.add_argument('--num_embeddings', type=int, default=1536)
    parser.add_argument("--train_ref_shot", type=int, default=4)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--residual_mode", type=str, default="sq", choices=["sq", "abs", "signed"])
    parser.add_argument("--match_mode", type=str, default="hard", choices=["hard", "soft_topk"])
    parser.add_argument("--match_topk", type=int, default=5)
    parser.add_argument("--match_tau", type=float, default=0.05)
    parser.add_argument("--match_chunk_size", type=int, default=8192)
    parser.add_argument("--use_raw_vqops", action="store_true")
    parser.add_argument("--raw_vq_pos", type=str, default="post_constraintor", choices=["pre_constraintor", "post_constraintor"])
    parser.add_argument("--raw_vq_debug", action="store_true")
    
    parser.add_argument("--use_wav", action="store_true")
    parser.add_argument("--wav_on", type=str, default="residual", choices=["residual", "feature"])
    parser.add_argument("--wave", type=str, default="haar", choices=["haar"])
    parser.add_argument("--hf_weight", type=float, default=1.0)
    parser.add_argument("--wav_mode", type=str, default="ll_hf", choices=["ll_hf", "ll_only", "skip_ll", "skip_hf", "hf_gate"])
    parser.add_argument("--feature_wav_mode", type=str, default="ll_only", choices=["ll_only", "hf_only", "ll_hf", "skip_ll", "skip_hf"])
    parser.add_argument("--ll_skip_alpha", type=float, default=0.5)
    parser.add_argument("--hf_gate_beta", type=float, default=1.0)
    parser.add_argument("--hf_skip_alpha", type=float, default=0.75)
    parser.add_argument("--wav_hf_normalize", action="store_true")
    parser.add_argument("--wav_shape_test", action="store_true")
    parser.add_argument("--dino_shape_test", action="store_true")
    parser.add_argument("--dinov2_feature_mode", type=str, default="final_projected", choices=DINOV2_FEATURE_MODES)
    parser.add_argument("--dinov2_layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--dinov2_proj_dim", type=int, default=256)
    parser.add_argument("--use_soft_codebook", action="store_true")
    parser.add_argument("--soft_cb_pos", type=str, default="post_constraintor", choices=["post_constraintor"])
    parser.add_argument("--soft_cb_k", type=int, default=512)
    parser.add_argument("--soft_cb_tau", type=float, default=0.2)
    parser.add_argument("--soft_cb_gamma", type=float, default=0.03)
    parser.add_argument("--soft_cb_warmup_epochs", type=int, default=5)
    parser.add_argument("--soft_cb_conf_gate", action="store_true")
    parser.add_argument("--soft_cb_gate_threshold", type=float, default=0.0)
    parser.add_argument("--soft_cb_gate_temp", type=float, default=0.05)
    
    args = parser.parse_args()
    init_seeds(42)
    if args.wav_shape_test:
        test_device = args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
        residual_wavelet_shape_test(device=test_device)
        print("Residual wavelet shape test passed.")
        raise SystemExit(0)
    if args.dino_shape_test:
        test_device = args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
        model_name = args.backbone if args.backbone in DINOV2_BACKBONES else "dinov2_vits14"
        dinov2_shape_test(
            model_name=model_name,
            device=test_device,
            feature_mode=args.dinov2_feature_mode,
            layers=args.dinov2_layers,
            proj_dim=args.dinov2_proj_dim,
        )
        print("DINOv2 shape test passed.")
        raise SystemExit(0)
    
    main(args)

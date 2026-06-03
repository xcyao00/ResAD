import os
import argparse
import numpy as np
from PIL import Image

import torch
import tqdm
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from models.fc_flow import load_flow_model

from datasets.mvtec import MVTEC
from datasets.visa import VISA
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS
from models.imagebind import ImageBindModel
from models.dinov2_backbone import DINOv2BackboneWrapper, DINOV2_BACKBONES, DINOV2_FEATURE_MODES
from models.dinov2_backbone import print_dinov2_config
from utils import load_weights

# ==========================================
# Haar Wavelet Filter の追加
# ==========================================
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
# ==========================================

class FEWSHOTDATA(Dataset):
    def __init__(self, 
                 root: str,
                 class_name: str = 'bottle', 
                 train: bool = True,
                 **kwargs) -> None:
    
        self.root = root
        self.class_name = class_name
        self.train = train
        self.mask_size = [kwargs.get('msk_crp_size'), kwargs.get('msk_crp_size')]
        
        self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)
    
        # set transforms
        self.transform = T.Compose([
            T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
            T.CenterCrop(kwargs.get('crp_size', 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size', 256), T.InterpolationMode.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size', 256)),
            T.ToTensor()])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label, mask_path, class_name = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx]
        img, label, mask = self._load_image_and_mask(image_path, label, mask_path)
        return img, label, mask, class_name
    
    def _load_image_and_mask(self, image_path, label, mask_path):
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            mask = torch.zeros([1, self.mask_size[0], self.mask_size[1]])
        else:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        return img, label, mask

    def _load_data(self, class_name):
        image_paths, labels, mask_paths = [], [], []
        phase = 'train' if self.train else 'test'
        
        image_dir = os.path.join(self.root, class_name, phase)
        mask_dir = os.path.join(self.root, class_name, 'ground_truth')

        img_types = sorted(os.listdir(image_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)])
            image_paths.extend(img_fpath_list)

            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                    
        class_names = [class_name] * len(image_paths)
        return image_paths, labels, mask_paths, class_names
    

SETTINGS = {'mvtec': MVTEC.CLASS_NAMES, 'visa': VISA.CLASS_NAMES,
            'btad': BTAD.CLASS_NAMES, 'mvtec3d': MVTEC3D.CLASS_NAMES,
            'mpdd': MPDD.CLASS_NAMES, 'mvtecloco': MVTECLOCO.CLASS_NAMES,
            'brats': BRATS.CLASS_NAMES}


def main(args):
    image_size = 224
    device = args.device
    root_dir = args.few_shot_dir
    # TODO: Consider adding a DINOv2-specific normalization option and compare it with the existing reference transform.
    if args.backbone == 'wide_resnet50_2':
        encoder = timm.create_model('wide_resnet50_2', features_only=True,
                out_indices=(1, 2, 3), pretrained=True).eval()
        encoder = encoder.to(device)
    elif args.backbone == 'tf_efficientnet_b6':
        encoder = timm.create_model('tf_efficientnet_b6', features_only=True,
                out_indices=(1, 2, 3), pretrained=True).eval()
        encoder = encoder.to(device)
    elif args.backbone in DINOV2_BACKBONES:
        encoder = DINOv2BackboneWrapper(
            model_name=args.backbone,
            out_dims=(40, 72, 200),
            out_sizes=(56, 28, 14),
            freeze=True,
            feature_mode=args.dinov2_feature_mode,
            layers=args.dinov2_layers,
            proj_dim=args.dinov2_proj_dim,
        ).to(device)
        encoder.eval()
        print_dinov2_config(encoder, image_size=image_size)
        
    # ウェーブレットフィルタの初期化
    wav_filter = HaarWaveletFilter(low_freq_weight=args.lf_weight, high_freq_weight=args.hf_weight).to(device)
    wav_filter.eval()
        
    feat_dims = encoder.feature_info.channels()    
    decoders = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    decoders = [decoder.to(args.device) for decoder in decoders]
    
    if args.bgadweight_dir:
        load_weights(encoder, decoders, args.bgadweight_dir)
    if args.dataset in SETTINGS.keys():
        CLASS_NAMES = SETTINGS[args.dataset]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.dataset}.")
    
    for class_name in CLASS_NAMES:
        train_dataset = FEWSHOTDATA(root_dir, class_name=class_name, train=True, img_size=image_size, crp_size=image_size,
                            msk_size=image_size, msk_crp_size=image_size)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False
        )
        layer1_features, layer2_features, layer3_features = [], [], []
        
        for batch in tqdm.tqdm(train_loader):
            images, _, _, _ = batch
            with torch.no_grad():
                patch_tokens = encoder(images.to(device))
                # 【追加】保存用に変換する前に、ウェーブレット変換をかける
                patch_tokens = [wav_filter(f) for f in patch_tokens]
                
            layer1_features.append(patch_tokens[0])
            layer2_features.append(patch_tokens[1])
            layer3_features.append(patch_tokens[2]) 
            
        layer1_features = torch.cat(layer1_features, dim=0)
        layer2_features = torch.cat(layer2_features, dim=0)
        layer3_features = torch.cat(layer3_features, dim=0)
        print(layer1_features.shape)
        print(layer2_features.shape)
        print(layer3_features.shape)
        
        layer1_channels = layer1_features.shape[1]
        layer2_channels = layer2_features.shape[1]
        layer3_channels = layer3_features.shape[1]

        layer1_features = layer1_features.permute(0, 2, 3, 1).reshape(-1, layer1_channels)
        layer2_features = layer2_features.permute(0, 2, 3, 1).reshape(-1, layer2_channels)
        layer3_features = layer3_features.permute(0, 2, 3, 1).reshape(-1, layer3_channels)
        
        os.makedirs(os.path.join(args.save_dir, class_name), exist_ok=True)
        
        print(f"Attempting to save layer1.npy for {class_name}...")
        np.save(os.path.join(args.save_dir, class_name, 'layer1.npy'), layer1_features.cpu().numpy())
        print(f"Successfully saved layer1.npy for {class_name}.")
        
        np.save(os.path.join(args.save_dir, class_name, 'layer2.npy'), layer2_features.cpu().numpy())
        np.save(os.path.join(args.save_dir, class_name, 'layer3.npy'), layer3_features.cpu().numpy())
        

def main2(args):
    # (既存のImageBind用の処理はそのまま残しています)
    image_size = 224
    device = args.device
    root_dir = args.few_shot_dir
    encoder = ImageBindModel(device=device)
    encoder.to(device)
    preprocess = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    
    if args.dataset in SETTINGS.keys():
        CLASS_NAMES = SETTINGS[args.dataset]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.dataset}.")
    
    for class_name in CLASS_NAMES:
        train_dataset = FEWSHOTDATA(root_dir, class_name=class_name, train=True, img_size=image_size, crp_size=image_size,
                            msk_size=image_size, msk_crp_size=image_size)
        train_dataset.transform = preprocess
        train_loader = DataLoader(
            train_dataset, batch_size=4, shuffle=False, num_workers=8, drop_last=False
        )
        layer1_features, layer2_features, layer3_features, layer4_features = [], [], [], []
        
        for batch in tqdm.tqdm(train_loader):
            images, _, _, _ = batch
            with torch.no_grad():
                patch_features = encoder.encode_image_from_tensors(images.to(device))
            layer1_features.append(patch_features[0])
            layer2_features.append(patch_features[1])
            layer3_features.append(patch_features[2]) 
            layer4_features.append(patch_features[3])
            
        layer1_features = torch.cat(layer1_features, dim=0)
        layer2_features = torch.cat(layer2_features, dim=0)
        layer3_features = torch.cat(layer3_features, dim=0)
        layer4_features = torch.cat(layer4_features, dim=0)
        print(layer1_features.shape)
        print(layer2_features.shape)
        print(layer3_features.shape)
        print(layer4_features.shape)
        
        layer1_features = layer1_features.reshape(-1, 1280)
        layer2_features = layer2_features.reshape(-1, 1280)
        layer3_features = layer3_features.reshape(-1, 1280)
        layer4_features = layer4_features.reshape(-1, 1280)
        
        os.makedirs(os.path.join(args.save_dir, class_name), exist_ok=True)
        
        np.save(os.path.join(args.save_dir, class_name, 'layer1.npy'), layer1_features.cpu().numpy())
        np.save(os.path.join(args.save_dir, class_name, 'layer2.npy'), layer2_features.cpu().numpy())
        np.save(os.path.join(args.save_dir, class_name, 'layer3.npy'), layer3_features.cpu().numpy())
        np.save(os.path.join(args.save_dir, class_name, 'layer4.npy'), layer4_features.cpu().numpy())
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mvtec")
    parser.add_argument('--few_shot_dir', type=str, default="./4shot/mvtec")
    parser.add_argument('--flow_arch', type=str, default='conditional_flow_model')
    parser.add_argument('--bgadweight_dir', type=str, default="")
    parser.add_argument('--save_dir', type=str, default="./ref_features/w50/mvtec_4shot_wav")
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")
    parser.add_argument("--dinov2_feature_mode", type=str, default="final_projected", choices=DINOV2_FEATURE_MODES)
    parser.add_argument("--dinov2_layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--dinov2_proj_dim", type=int, default=256)
    parser.add_argument('--coupling_layers', type=int, default=10)
    parser.add_argument('--clamp_alpha', type=float, default=1.9)
    parser.add_argument('--pos_embed_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default="cuda:0")
    
    # 追加: ウェーブレット変換のパラメータ
    parser.add_argument("--lf_weight", type=float, default=0.1, help="Weight for low frequency (LL) components")
    parser.add_argument("--hf_weight", type=float, default=1.2, help="Weight for high frequency (LH, HL, HH) components")
    
    args = parser.parse_args()
    main(args)

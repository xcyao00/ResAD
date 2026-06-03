#wideresnetの特徴抽出後に少し加工
import os
import argparse
import numpy as np
from PIL import Image

import torch
import tqdm
import timm
import torch.nn as nn
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
from utils import load_weights
import torch.nn.functional as F

class WideResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name='wide_resnet50_2', out_indices=(1, 2, 3)):
        super().__init__()
        # 元のモデルを読み込み
        self.encoder = timm.create_model(model_name, features_only=True, out_indices=out_indices, pretrained=True)
        self.embed_dims = self.encoder.feature_info.channels()
        
        # 重みを持たない平滑化レイヤー
        self.layer1_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # 浅い層用
        self.layer3_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)  # 深い層用（より広範囲をぼかす）

    def forward(self, x):
        features = self.encoder(x)
        processed_features = []
        
        for i, feat in enumerate(features):
            if i == 0:
                # 浅い層: 局所的な細かい変動ノイズを吸収
                feat = self.layer1_pool(feat)
            elif i == 2:
                # 深い層: 意味的な情報を少し広げてマッチングを安定させる
                feat = self.layer3_pool(feat)
            
            # 全層共通: ベクトルのスケールを統一し、次元が大きくても距離計算を安定させる
            feat = F.normalize(feat, p=2, dim=1)
            processed_features.append(feat)
            
        return processed_features
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
            # load images
            img_type_dir = os.path.join(image_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)])
            image_paths.extend(img_fpath_list)

            # load gt labels
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
    device = 'cuda:0'
    root_dir = args.few_shot_dir
    if args.backbone == 'wide_resnet50_2':
        encoder = WideResNetFeatureExtractor(model_name='wide_resnet50_2', out_indices=(1, 2, 3)).eval()
        encoder = encoder.to(device)
        feat_dims = encoder.embed_dims
    elif args.backbone == 'tf_efficientnet_b6':#10/26追加
        encoder = timm.create_model('tf_efficientnet_b6', features_only=True,
                out_indices=(1, 2, 3), pretrained=True).eval()  # the pretrained checkpoint will be in /home/.cache/torch/hub/checkpoints/
        encoder = encoder.to(device)
    #feat_dims = encoder.feature_info.channels()    
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
            layer1_features.append(patch_tokens[0])
            layer2_features.append(patch_tokens[1])
            layer3_features.append(patch_tokens[2]) 
        layer1_features = torch.cat(layer1_features, dim=0)
        layer2_features = torch.cat(layer2_features, dim=0)
        layer3_features = torch.cat(layer3_features, dim=0)
        print(layer1_features.shape)
        print(layer2_features.shape)
        print(layer3_features.shape)
        #修正10/26
        layer1_channels = layer1_features.shape[1]
        layer2_channels = layer2_features.shape[1]
        layer3_channels = layer3_features.shape[1]

        layer1_features = layer1_features.permute(0, 2, 3, 1).reshape(-1, layer1_channels)
        layer2_features = layer2_features.permute(0, 2, 3, 1).reshape(-1, layer2_channels)
        layer3_features = layer3_features.permute(0, 2, 3, 1).reshape(-1, layer3_channels)
        #修正終わり
        os.makedirs(os.path.join(args.save_dir, class_name), exist_ok=True)
        
        print(f"Attempting to save layer1.npy for {class_name}...")
        np.save(os.path.join(args.save_dir, class_name, 'layer1.npy'), layer1_features.cpu().numpy())
        print(f"Successfully saved layer1.npy for {class_name}.")
        
        np.save(os.path.join(args.save_dir, class_name, 'layer2.npy'), layer2_features.cpu().numpy())
        np.save(os.path.join(args.save_dir, class_name, 'layer3.npy'), layer3_features.cpu().numpy())
        

def main2(args):
    image_size = 224
    device = 'cuda:0'
    root_dir = args.few_shot_dir
    encoder = ImageBindModel(device=device)
    encoder.to(device)
    preprocess = T.Compose(  # for imagebind
            [
                T.Resize(
                    image_size, interpolation=T.InterpolationMode.BICUBIC
                ),
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
    parser.add_argument('--bgadweight_dir', type=str, default="")# 12/16追加
    parser.add_argument('--save_dir', type=str, default="./ref_features/w50/mvtec_4shot")
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")#10/26追加
    parser.add_argument('--coupling_layers', type=int, default=10)
    parser.add_argument('--clamp_alpha', type=float, default=1.9)
    parser.add_argument('--pos_embed_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)

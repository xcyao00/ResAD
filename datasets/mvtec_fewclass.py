"""
The dataset defined in this script is only used for cross-class training,
where we use both normal and abnormal samples for training. And we use all
abnormal samples from the test set, as these abnormal samples will not be
tested in cross-class setting.
"""
import os
import random
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.transforms import RandomHorizontalFlip
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T

import cv2
import glob
import imgaug.augmenters as iaa
import albumentations as A


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTECFEWANO(Dataset):
    """This dataset is used for cross-class training, where we use all the normal and abnomal
    samples for training. As we will not test on the training classes, using abnormal samples
    in test set is actually reasonable.

    Args:
        root (string): Root directory of dataset, i.e ``../../mvtec_anomaly_detection``.
        train (bool, optional): If True, creates dataset for training, otherwise for testing.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.Resize``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    MVTEC_URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
    
    CLASS_NAMES = ['capsule','screw','transistor']

    def __init__(
            self, 
            root: str,
            class_name: str,
            train: bool = True,
            normalize: str = 'imagebind',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            **kwargs):
        
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('msk_crp_size'), kwargs.get('msk_crp_size')]
        
        # load dataset
        if isinstance(self.class_name, str):
            self.image_paths, self.labels, self.mask_paths, self.class_names,self.anomaly_types = self._load_data(self.class_name)
        elif self.class_name is None:
            self.image_paths, self.labels, self.mask_paths, self.class_names,self.anomaly_types = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.class_names,self.anomaly_types = self._load_all_data(self.class_name)
            
        if normalize == "imagebind":
            self.transform = T.Compose(  # for imagebind
                [
                    T.Resize(
                        224, interpolation=T.InterpolationMode.BICUBIC
                    ),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        else:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
                T.CenterCrop(kwargs.get('crp_size', 224)),
                T.ToTensor(),
                T.Compose([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])])
        
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size'), Image.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size')),
            T.ToTensor()])
            
        self.class_to_idx = {'capsule': 0,'screw': 1, 'toothbrush': 2}
        self.idx_to_class = { 0: 'capsule', 1: 'screw', 2: 'transistor'}
    
    def __getitem__(self, idx):
        image_path, label, mask_path, class_name,anomaly_type = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx],self.anomaly_types[idx]
        img, label, mask = self._load_image_and_mask(image_path, label, mask_path)
        
        return img, label, mask, class_name,anomaly_type
    
    def _load_image_and_mask(self, image_path, label, mask_path):
        img = Image.open(image_path).convert('RGB')
        class_name = image_path.split('/')[-4]
        # if class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
        #     img = np.expand_dims(np.asarray(img), axis=2)
        #     img = np.concatenate([img, img, img], axis=2)
        #     img = Image.fromarray(img.astype('uint8')).convert('RGB')
        #
        img = self.transform(img)
        #
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        
        return img, label, mask

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self, class_name):
        image_paths, labels, mask_paths,anomaly_types = [], [], [], []
        class_names_list = []  # Initialize class_names_list here

        
        for phase in ['train', 'test']:
            image_dir = os.path.join(self.root, class_name, phase)
            mask_dir = os.path.join(self.root, class_name, 'ground_truth')

            img_types = sorted(os.listdir(image_dir))
            for img_type in img_types:
                # load images
                img_type_dir = os.path.join(image_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                        for f in os.listdir(img_type_dir)
                                        if f.endswith('.png')])
                image_paths.extend(img_fpath_list)

                # load gt labels
                if img_type == 'good':
                    labels.extend([0] * len(img_fpath_list))
                    mask_paths.extend([None] * len(img_fpath_list))
                    anomaly_types.extend(['good'] * len(img_fpath_list)) # 'good'を追加
                else:
                    labels.extend([1] * len(img_fpath_list))
                    gt_type_dir = os.path.join(mask_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                    for img_fname in img_fname_list]
                    mask_paths.extend(gt_fpath_list)
                    anomaly_types.extend([img_type] * len(img_fpath_list)) # 異常タイプ名を追加
                    
        class_names_list = [class_name] * len(image_paths) # 変数名が衝突しないように変更
        return image_paths, labels, mask_paths, class_names_list, anomaly_types # anomaly_types も返す
    
    def _load_all_data(self, class_names=None):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_class_names = []
        all_anomaly_types = [] # anomaly_types を追加
        CLASS_NAMES = class_names if class_names is not None else self.CLASS_NAMES
        for class_name in CLASS_NAMES:
            image_paths, labels, mask_paths, class_names,anomaly_types_from_load_data = self._load_data(class_name)
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_class_names.extend(class_names)
            all_anomaly_types.extend(anomaly_types_from_load_data) # anomaly_types を追加
        return all_image_paths, all_labels, all_mask_paths, all_class_names, all_anomaly_types # anomaly_types も返す


class MVTECFEW(Dataset):
    
    CLASS_NAMES = ['capsule','screw','transistor']
    def __init__(self, 
                 root: str,
                 class_name: str = 'capsule', 
                 train: bool = True,
                 normalize: str = 'imagebind',
                 **kwargs) -> None:
    
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('msk_crp_size'), kwargs.get('msk_crp_size')]
        
        if isinstance(self.class_name, str):
            self.image_paths, self.labels, self.mask_paths, self.class_names,self.anomaly_types = self._load_data(self.class_name)
        elif self.class_name is None:
            self.image_paths, self.labels, self.mask_paths, self.class_names,self.anomaly_types = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.class_names,self.anomaly_types = self._load_all_data(self.class_name)
        
        # set transforms
        if normalize == "imagebind":
            self.transform = T.Compose(  # for imagebind
                [
                    T.Resize(
                        224, interpolation=T.InterpolationMode.BICUBIC
                    ),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        else:
            self.transform = T.Compose([
                T.Resize(kwargs.get('img_size', 224), T.InterpolationMode.BICUBIC),
                T.CenterCrop(kwargs.get('crp_size', 224)),
                T.ToTensor(),
                T.Compose([T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])])
        
        # mask
        self.target_transform = T.Compose([
            T.Resize(kwargs.get('msk_size', 224), Image.NEAREST),
            T.CenterCrop(kwargs.get('msk_crp_size', 224)),
            T.ToTensor()])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label, mask_path, class_name,anomaly_type = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx], self.anomaly_types[idx] # anomaly_type を追加
        img, label, mask = self._load_image_and_mask(image_path, label, mask_path)
        
        return img, label, mask, class_name,anomaly_type # anomaly_type を返す
    
    def _load_image_and_mask(self, image_path, label, mask_path):
        img = Image.open(image_path).convert('RGB')
        class_name = image_path.split('/')[-4]
        # if class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
        #     img = np.expand_dims(np.asarray(img), axis=2)
        #     img = np.concatenate([img, img, img], axis=2)
        #     img = Image.fromarray(img.astype('uint8')).convert('RGB')
        #
        img = self.transform(img)
        #
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask_path)
            mask = self.target_transform(mask)
        
        return img, label, mask

    def _load_data(self, class_name):
        image_paths, labels, mask_paths,anomaly_types = [], [], [], []
        class_names_list = []  # Initialize class_names_list here

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
                                    for f in os.listdir(img_type_dir)
                                    if f.endswith('.png')])
            image_paths.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                labels.extend([0] * len(img_fpath_list))
                mask_paths.extend([None] * len(img_fpath_list))
                anomaly_types.extend(['good'] * len(img_fpath_list)) # 'good'を追加
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(mask_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                for img_fname in img_fname_list]
                mask_paths.extend(gt_fpath_list)
                anomaly_types.extend([img_type] * len(img_fpath_list)) # 異常タイプ名を追加
                    
        class_names_list = [class_name] * len(image_paths) # 変数名が衝突しないように変更
        return image_paths, labels, mask_paths, class_names_list, anomaly_types # anomaly_types も返す
    
    def _load_all_data(self, class_names=None):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_class_names = []
        all_anomaly_types = [] # anomaly_types を追加
        CLASS_NAMES = class_names if class_names is not None else self.CLASS_NAMES
        for class_name in CLASS_NAMES:
            image_paths, labels, mask_paths, class_names, anomaly_types_from_load_data = self._load_data(class_name)
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_class_names.extend(class_names)
            all_anomaly_types.extend(anomaly_types_from_load_data) # anomaly_types を追加
        return all_image_paths, all_labels, all_mask_paths, all_class_names, all_anomaly_types # anomaly_types も返す


def get_normal_image_paths_mvtec(root, class_name):
    phase = 'train' 
    image_paths = []

    image_dir = os.path.join(root, class_name, phase)
    
    img_types = sorted(os.listdir(image_dir))
    for img_type in img_types:
        # load images
        img_type_dir = os.path.join(image_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                    for f in os.listdir(img_type_dir)])
        image_paths.extend(img_fpath_list)

    return image_paths

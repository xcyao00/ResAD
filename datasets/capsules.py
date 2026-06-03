import os
import pandas
import torch
import numpy as np
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.transforms import RandomHorizontalFlip


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CAPSULES(Dataset):
    
    CLASS_NAMES = ['capsules']
    
    def __init__(self, 
                 root: str,
                 class_name: str,
                 train: bool = True,
                 normalize: str = 'imagebind',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs):
    
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('crp_size'), kwargs.get('crp_size')]
        
        # load dataset
        if isinstance(self.class_name, str):
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)
        elif self.class_name is None:  # load all classes
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data(self.class_name)
        
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
            T.Resize(kwargs.get('img_size'), Image.NEAREST),
            T.CenterCrop(kwargs.get('crp_size')),
            T.ToTensor()])
        
        self.class_to_idx = {'capsules': 0 }
        self.idx_to_class = {0:'capsules' }

    def __getitem__(self, idx):
        image_path, label, mask, class_name = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx]
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = np.array(mask)
            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)
        
        if self.train:
            label = self.class_to_idx[class_name]
        
        return image, label, mask, class_name

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self, class_name):
        split_csv_file = os.path.join(self.root, 'split_csv', '1cls.csv')
        csv_data = pandas.read_csv(split_csv_file)
        
        class_data = csv_data.loc[csv_data['object'] == class_name]
        
        if self.train:
            train_data = class_data.loc[class_data['split'] == 'train']
            image_paths = train_data['image'].to_list()
            image_paths = [os.path.join(self.root, file_name) for file_name in image_paths]
            labels = [0] * len(image_paths)
            mask_paths = [None] * len(image_paths)
        else:
            image_paths, labels, mask_paths = [], [], []
            
            test_data = class_data.loc[class_data['split'] == 'test']
            test_normal_data = test_data.loc[test_data['label'] == 'normal']
            test_anomaly_data = test_data.loc[test_data['label'] == 'anomaly']
            
            normal_image_paths = test_normal_data['image'].to_list()
            normal_image_paths = [os.path.join(self.root, file_name) for file_name in normal_image_paths]
            image_paths.extend(normal_image_paths)
            labels.extend([0] * len(normal_image_paths))
            mask_paths.extend([None] * len(normal_image_paths))
            
            anomaly_image_paths = test_anomaly_data['image'].to_list()
            anomaly_mask_paths = test_anomaly_data['mask'].to_list()
            anomaly_image_paths = [os.path.join(self.root, file_name) for file_name in anomaly_image_paths]
            anomaly_mask_paths = [os.path.join(self.root, file_name) for file_name in anomaly_mask_paths]
            image_paths.extend(anomaly_image_paths)
            labels.extend([1] * len(anomaly_image_paths))
            mask_paths.extend(anomaly_mask_paths)

        class_names = [class_name] * len(image_paths)
        return image_paths, labels, mask_paths, class_names
    
    def _load_all_data(self, class_names=None):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_class_names = []
        CLASS_NAMES = class_names if class_names is not None else self.CLASS_NAMES
        for class_name in CLASS_NAMES:
            image_paths, labels, mask_paths, class_names = self._load_data(class_name)
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_class_names.extend(class_names)
        return all_image_paths, all_labels, all_mask_paths, all_class_names

    def update_class_to_idx(self, class_to_idx):
        for class_name in self.class_to_idx.keys():
            self.class_to_idx[class_name] = class_to_idx[class_name]
        class_names = self.class_to_idx.keys()
        idxs = self.class_to_idx.values()
        self.idx_to_class = dict(zip(idxs, class_names))


class CAPSULESANO(Dataset):
    
    CLASS_NAMES = ['capsules']
    
    def __init__(self, 
                 root: str,
                 class_name: str,
                 train: bool = True,
                 normalize: str = 'imagebind',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs):
        
        self.root = root
        self.class_name = class_name
        self.train = train
        self.cropsize = [kwargs.get('crp_size'), kwargs.get('crp_size')]
        
        # load dataset
        if isinstance(self.class_name, str):
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_data(self.class_name)
        elif self.class_name is None:
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data()
        else:
            self.image_paths, self.labels, self.mask_paths, self.class_names = self._load_all_data(self.class_name)
        
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
            T.Resize(kwargs.get('img_size'), Image.NEAREST),
            T.CenterCrop(kwargs.get('crp_size')),
            T.ToTensor()])
        
        self.class_to_idx = {'capsules':0}
        self.idx_to_class = {0:'capsules'}

    def __getitem__(self, idx):
        image_path, label, mask, class_name = self.image_paths[idx], self.labels[idx], self.mask_paths[idx], self.class_names[idx]
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = np.array(mask)
            mask[mask != 0] = 255
            mask = Image.fromarray(mask)
            mask = self.target_transform(mask)
        
        if self.train:
            label = self.class_to_idx[class_name]
        
        return image, label, mask, class_name

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self, class_name):
        split_csv_file = os.path.join(self.root, 'split_csv', '1cls.csv')
        csv_data = pandas.read_csv(split_csv_file)
        
        class_data = csv_data.loc[csv_data['object'] == class_name]
        all_image_paths, all_labels, all_mask_paths = [], [], []
        
        # train
        train_data = class_data.loc[class_data['split'] == 'train']
        image_paths = train_data['image'].to_list()
        image_paths = [os.path.join(self.root, file_name) for file_name in image_paths]
        labels = [0] * len(image_paths)
        mask_paths = [None] * len(image_paths)
        all_image_paths.extend(image_paths)
        all_labels.extend(labels)
        all_mask_paths.extend(mask_paths)
        
        # test 
        image_paths, labels, mask_paths = [], [], []
        test_data = class_data.loc[class_data['split'] == 'test']
        test_normal_data = test_data.loc[test_data['label'] == 'normal']
        test_anomaly_data = test_data.loc[test_data['label'] == 'anomaly']
        
        normal_image_paths = test_normal_data['image'].to_list()
        normal_image_paths = [os.path.join(self.root, file_name) for file_name in normal_image_paths]
        image_paths.extend(normal_image_paths)
        labels.extend([0] * len(normal_image_paths))
        mask_paths.extend([None] * len(normal_image_paths))
        
        anomaly_image_paths = test_anomaly_data['image'].to_list()
        anomaly_mask_paths = test_anomaly_data['mask'].to_list()
        anomaly_image_paths = [os.path.join(self.root, file_name) for file_name in anomaly_image_paths]
        anomaly_mask_paths = [os.path.join(self.root, file_name) for file_name in anomaly_mask_paths]
        image_paths.extend(anomaly_image_paths)
        labels.extend([1] * len(anomaly_image_paths))
        mask_paths.extend(anomaly_mask_paths)
        
        all_image_paths.extend(image_paths)
        all_labels.extend(labels)
        all_mask_paths.extend(mask_paths)

        class_names = [class_name] * len(all_image_paths)
        return all_image_paths, all_labels, all_mask_paths, class_names
    
    def _load_all_data(self, class_names=None):
        all_image_paths = []
        all_labels = []
        all_mask_paths = []
        all_class_names = []
        CLASS_NAMES = class_names if class_names is not None else self.CLASS_NAMES
        for class_name in CLASS_NAMES:
            image_paths, labels, mask_paths, class_names = self._load_data(class_name)
            all_image_paths.extend(image_paths)
            all_labels.extend(labels)
            all_mask_paths.extend(mask_paths)
            all_class_names.extend(class_names)
        return all_image_paths, all_labels, all_mask_paths, all_class_names

    def update_class_to_idx(self, class_to_idx):
        for class_name in self.class_to_idx.keys():
            self.class_to_idx[class_name] = class_to_idx[class_name]
        class_names = self.class_to_idx.keys()
        idxs = self.class_to_idx.values()
        self.idx_to_class = dict(zip(idxs, class_names))


def get_normal_image_paths_visa(root, class_name):
    split_csv_file = os.path.join(root, 'split_csv', '1cls.csv')
    csv_data = pandas.read_csv(split_csv_file)
    
    class_data = csv_data.loc[csv_data['object'] == class_name]
    
    train_data = class_data.loc[class_data['split'] == 'train']
    image_paths = train_data['image'].to_list()
    image_paths = [os.path.join(root, file_name) for file_name in image_paths]

    return image_paths

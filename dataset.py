import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': TF.to_tensor(image), 'label': TF.to_tensor(label)}

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {
            'image': TF.resize(image, self.size), 
            
            'label': TF.resize(label, self.size, interpolation=InterpolationMode.NEAREST)
        }

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': TF.hflip(image), 'label': TF.hflip(label)}
        return {'image': image, 'label': label}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': TF.vflip(image), 'label': TF.vflip(label)}
        return {'image': image, 'label': label}

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = TF.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode):
        # 使用 os.path.join 彻底解决路径拼接报错
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')
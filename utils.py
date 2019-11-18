import os
import cv2
import torch
import math

import numpy as np
import albumentations as albu
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, sampler
from albumentations import (Normalize, Compose)
from albumentations.torch import ToTensor
from albumentations import (HorizontalFlip, VerticalFlip, IAAPiecewiseAffine, ElasticTransform, IAASharpen, RandomBrightness,  ShiftScaleRotate, OneOf, Flip, Normalize, Resize, Compose, GaussNoise, ElasticTransform)
from albumentations.imgaug.transforms import IAASharpen
from albumentations.torch import ToTensor
from sklearn.model_selection import StratifiedKFold
from torch.optim.optimizer import Optimizer, required


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
    
    
class RSNADataset(Dataset):
    
    def __init__(self, df, data_folder, transforms):
        self.df = df
        self.root = data_folder
        self.transforms = transforms
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = df_row['name']+'.png'
        image_path = os.path.join(self.root, image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img)
        img = augmented['image']
        label = np.array(df_row[df_row.keys()[1:7]].tolist(),dtype=float)
        return img, torch.from_numpy(label).float()

    def __len__(self):
        return len(self.fnames)

    
class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),TTA=False):
        self.root = root
        df['imageid'] = df['ID'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1] + '.png')
        self.fnames = df['imageid'].unique().tolist()
        self.num_samples = len(self.fnames)
        if TTA==True:
            self.transform = Compose(
                [
                    HorizontalFlip(),
                    albu.Resize(512, 512),
                    Normalize(mean=mean,std=std,p=1),
                    ToTensor(),
                ]
            )
        else:
            self.transform = Compose(
                [
                    albu.Resize(512, 512),
                    Normalize(mean=mean, std=std, p=1),
                    ToTensor(),
                ]
            )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples

    
def make_transforms(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    list_transforms = []
    if phase == "train":
         list_transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.Rotate(limit=15),
                albu.RandomContrast(limit=0.2),
                albu.RandomBrightness(limit=0.2)
            ]
         )
    list_transforms.extend(
        [
            albu.Resize(512, 512),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms
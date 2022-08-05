# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import math
import random
import numpy as np

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=8, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return mask


class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGBA') if img.mode != 'RGBA' else img),
            T.CenterCrop((192,192)),
            #T.RandomHorizontalFlip(),
            T.ToTensor(),
            #T.Normalize(mean=torch.tensor([0.5073, 0.5073, 0.5073, 1]),std=torch.tensor([0.2534, 0.2534, 0.2534, 0.1])),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        elif config.MODEL.TYPE == 'SwinRec':
            model_patch_size=config.MODEL.SWINREC.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(config, logger, train):
    if train:
        transform = SimMIMTransform(config)
        logger.info(f'Pre-train data transform:\n{transform}')

        dataset = ImageFolder(config.DATA.DATA_PATH+'/train', transform)
        logger.info(f'Build dataset: train images = {len(dataset)}')
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
	    
        return dataloader
    else:
        transform = SimMIMTransform(config)
        logger.info(f'Pre-train data transform:\n{transform}')

        dataset = ImageFolder(config.DATA.DATA_PATH+'/test', transform)
        logger.info(f'Build dataset: train images = {len(dataset)}')
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
        dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)
	    
        return dataloader

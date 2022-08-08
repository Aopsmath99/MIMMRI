# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# Modified by Kyler Larsen
# --------------------------------------------------------

from functools import partial
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer
from .SwinRecNew import SwinRecNet
from .SwinRec import *
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from fastmri.data import transforms as T
import fastmri
import PIL
import torchvision.transforms as Tr
import torchvision.transforms as transforms
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
import math
import torch.nn.functional as F

class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask, target):

        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)

        w = mask.squeeze(0).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)

        return x




class SwinRecTransformerForSimMIM(SwinRecNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.squeeze(0).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 4, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask, target, loss_array):
        firsts = True
        firstl = True
        #z = self.encoder(x, mask, target)
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        x = x.squeeze()
        x_rec = x_rec.squeeze()
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x.float(), x_rec.float(), reduction='none')

        #i = random.randint(0, 10000000)
        transform = Tr.ToPILImage()
        x_rec = transform(x_rec)
        x = transform(x)
        loss = (loss_recon.cuda() * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        #Saving quality images for the purpose of image gathering
        '''
        if loss.item() < 0.05:
            if firstl: 
                x_rec.save('/rfanfs/pnl-zorro/home/kyler/SimMIM/output_images_norm/goodrec_kspace' + str(i) + '.png')
                x.save('/rfanfs/pnl-zorro/home/kyler/SimMIM/output_images_norm/original_kspace' + str(i) + '.png')
                firstl = False
            elif i < 5000000:
                x_rec.save('/rfanfs/pnl-zorro/home/kyler/SimMIM/output_images_norm/goodrec_kspace' + str(i) + '.png')
                x.save('/rfanfs/pnl-zorro/home/kyler/SimMIM/output_images_norm/original_kspace' + str(i) + '.png')
        '''   
        a = ImageOps.grayscale(x)
        b = ImageOps.grayscale(x_rec)
        a = img_as_float(a)
        b = img_as_float(b)
        (score, diff) = ssim(a, b, full=True)
        if score > 0.995:
            if firsts:
                x_rec.save('/rfanfs/pnl-zorro/home/kyler/SimMIM/vit_images/goodscorerec5_kspace' + str(i) + '.png')
                x.save('/rfanfs/pnl-zorro/home/kyler/SimMIM/vit_images/originalscore5_kspace' + str(i) + '.png')
                firsts = False
            elif i < 500000:
                x_rec.save('/rfanfs/pnl-zorro/home/kyler/SimMIM/vit_images/goodscorerec5_kspace' + str(i) + '.png')
                x.save('/rfanfs/pnl-zorro/home/kyler/SimMIM/vit_images/originalscore5_kspace' + str(i) + '.png')
            print("THIS IS A GOOD ONE, SCORE: ", score)
        return loss, score

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    print(config)
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
    elif model_type == 'SwinRec':
        encoder = SwinRecNet(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWINREC.PATCH_SIZE,
            in_chans=config.MODEL.SWINREC.IN_CHANS,
            embed_dim=config.MODEL.SWINREC.EMBED_DIM,
            depths=config.MODEL.SWINREC.DEPTHS,
            num_heads=config.MODEL.SWINREC.NUM_HEADS,
            window_size=config.MODEL.SWINREC.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWINREC.MLP_RATIO,
            qkv_bias=config.MODEL.SWINREC.QKV_BIAS,
            qk_scale=config.MODEL.SWINREC.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWINREC.APE,
            patch_norm=config.MODEL.SWINREC.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
            upscale = config.MODEL.SWINREC.UPSCALE,
            img_range = config.MODEL.SWINREC.IMG_RANGE,
            upsampler = config.MODEL.SWINREC.UPSAMPLER,
            resi_connection = config.MODEL.SWINREC.RESI_CONNECTION)
        encoder_stride = 32
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    return model

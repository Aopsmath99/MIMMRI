# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# Further modified by Kyler Larsen
# --------------------------------------------------------

from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .SwinRec import build_SwinRec
from .simmim import build_simmim


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_simmim(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        elif model_type == 'SwinRec':
            model = build_SwinRec(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")
    return model

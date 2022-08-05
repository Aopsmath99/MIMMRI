from .data_simmim import build_loader_simmim
from .data_finetune import build_loader_finetune

def build_loader(config, logger, is_pretrain, train):
    if is_pretrain:
        return build_loader_simmim(config, logger, train)
    else:
        return build_loader_finetune(config, logger, train)

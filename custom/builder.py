from registry import Registry, build_from_cfg
from dataset import CustomDataset

MODELS = Registry('model')
BACKBONES = Registry('backbone')
    
    
def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONES)

def build_model(cfg_model):
    mask_rcnn = build_from_cfg(cfg_model, MODELS)
    print("?!")
    exit()

def build_dataset(cfg):
    return CustomDataset(**cfg)
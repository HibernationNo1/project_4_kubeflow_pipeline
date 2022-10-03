from registry import Registry, build_from_cfg
from dataset import CustomDataset


MODELS = Registry('model')
BACKBONES = Registry('backbone')

def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONES)

def build_model(cfg_model):
    
    assert cfg_model.get('backbone') is not None , 'backbone must be specified in model field'
    build_backbone(cfg_model.backbone)
    # return build_from_cfg(cfg_model)
    

def build_dataset(cfg):
    return CustomDataset(**cfg)
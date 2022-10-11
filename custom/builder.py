from functools import partial
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from utils.utils import collate
from utils.registry import Registry, build_from_cfg
from utils.sampler import GroupSampler
from dataset import CustomDataset

MODELS = Registry('model')
BACKBONES = Registry('backbone')
    
    
def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONES)

def build_model(cfg_model):
    mask_rcnn = build_from_cfg(cfg_model, MODELS)
    return mask_rcnn

def build_dataset(cfg):
    return CustomDataset(**cfg)

def build_dataloader(dataset,
                     batch_size,
                     num_workers,
                     seed,
                     shuffle = True
                    ):

    sampler = GroupSampler(dataset, batch_size) if shuffle else None
    batch_sampler = None
    
    init_fn = partial(worker_init_fn, num_workers=num_workers,seed=seed) if seed is not None else None
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=partial(collate, samples_per_gpu=batch_size),    
        pin_memory=False,
        worker_init_fn=init_fn) # TODO : run에서 persistent_workers사용하는지 확인
    
    return data_loader
    
    
def worker_init_fn(worker_id, num_workers, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    


def build_dp():
import numpy as np
import random
import copy
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from functools import partial
from itertools import chain

from utils.utils import collate, auto_scale_lr
from utils.registry import Registry, build_from_cfg
from utils.sampler import GroupSampler
from utils.optimizer import DefaultOptimizerConstructor
from utils.runner import EpochBasedRunner
from utils.scatter import scatter_kwargs
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


def build_runner(cfg: dict):
    runner_cfg = copy.deepcopy(cfg)
    runner = EpochBasedRunner(**runner_cfg)
 
    return runner

def build_optimizer(model, whole_cfg, logger):
    auto_scale_lr(whole_cfg, logger)
    cfg = whole_cfg.optimizer
    optimizer_cfg = copy.deepcopy(cfg)
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    
    Constructor_cfg = dict(optimizer_cfg=optimizer_cfg,
                           paramwise_cfg=paramwise_cfg)
    optim_constructor = DefaultOptimizerConstructor(**Constructor_cfg)
    
    optimizer = optim_constructor(model)
    return optimizer



def build_dp(model, device='cuda', dim=0):
    if device == 'cuda': 
        model = model.cuda()
    
    return MMDataParallel(model, dim=dim)
    
    
class MMDataParallel(DataParallel):
    """The DataParallel module that supports DataContainer.

    MMDataParallel has two main differences with PyTorch DataParallel:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data during both GPU and CPU inference.
    - It implement two more APIs ``train_step()`` and ``val_step()``.

    .. warning::
        MMDataParallel only supports single GPU training, if you need to
        train with multiple GPUs, please use MMDistributedDataParallel
        instead. If you have multiple GPUs and you just want to use
        MMDataParallel, you can set the environment variable
        ``CUDA_VISIBLE_DEVICES=0`` or instantiate ``MMDataParallel`` with
        ``device_ids=[0]``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        device_ids (list[int]): Device IDS of modules to be scattered to.
            Defaults to None when GPU is not available.
        output_device (str | int): Device ID for output. Defaults to None.
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """
    def __init__(self, model, dim=0, device_ids = [0]):
        super().__init__(model, dim=dim, device_ids = device_ids)
        self.dim = dim
        
        
    def train_step(self, *inputs, **kwargs):
        assert self.device_ids == [0], "this project is for only single gpu with ID == '0',\
                                        but device_ids is {self.device_ids}"
        
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        
        
        inputs, kwargs = scatter_kwargs(inputs, kwargs, self.device_ids)
        return self.module.train_step(*inputs[0], **kwargs[0])
    
    
    

    


    


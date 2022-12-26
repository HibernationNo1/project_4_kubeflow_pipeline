from collections import OrderedDict
import copy
import torch
from torch.nn.parallel import DataParallel
from itertools import chain

from utils.utils import auto_scale_lr
from utils.config import CONFIGDICT_NAME
from utils.registry import Registry, build_from_cfg
from utils.optimizer import DefaultOptimizerConstructor
from utils.runner import EpochBasedRunner
from utils.scatter import scatter_inputs
from datasets.dataset import CustomDataset
from datasets.dataloader import _build_dataloader

MODELS = Registry('model')
BACKBONES = Registry('backbone')


    
    
def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONES)

def build_model(model_cfg):

    
    if model_cfg._class_name != CONFIGDICT_NAME:
        raise TypeError(f"config must be class: ConfigDict, but got {type(model_cfg)}")
    mask_rcnn = build_from_cfg(model_cfg, MODELS)

    return mask_rcnn

def _build_dataset(dataset_cfg):
    if dataset_cfg is None: return None
    else: return CustomDataset(**dataset_cfg)
        
def build_dataset(train_cfg = None, val_cfg = None):
    train_dataset = _build_dataset(train_cfg)
    val_dataset = _build_dataset(val_cfg)    

    if train_dataset is not None and val_dataset is None:   return train_dataset, None        # only train dataset
    elif val_dataset is not None and train_dataset is None: return None, val_dataset          # only val dataset
    else: return train_dataset, val_dataset     

def build_dataloader(num_workers, seed,
                     train_dataset=None, train_batch_size=None,
                     val_dataset=None, val_batch_size=None,
                     shuffle = True):
    train_dataloader = _build_dataloader(train_dataset, train_batch_size, num_workers, seed, shuffle = shuffle)
    val_dataloader = _build_dataloader(val_dataset, val_batch_size, num_workers, seed, shuffle = shuffle)
    
    if train_dataloader is not None and val_dataloader is None: return train_dataloader, None        # only train dataset
    elif val_dataloader is not None and train_dataloader is None: return None, val_dataloader          # only val dataset

    return train_dataloader, val_dataloader


def build_runner(cfg: dict):
    runner_cfg = copy.deepcopy(cfg)
    runner = EpochBasedRunner(**runner_cfg)
 
    return runner

def build_optimizer(model, whole_cfg, logger):
    auto_scale_lr(whole_cfg, logger)
    optimizer_cfg = copy.deepcopy(whole_cfg.optimizer)
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    
    Constructor_cfg = dict(optimizer_cfg=optimizer_cfg,
                           paramwise_cfg=paramwise_cfg)
    
    optim_constructor = DefaultOptimizerConstructor(**Constructor_cfg)
    
    optimizer = optim_constructor(model)
    return optimizer



def build_dp(model, cfg, device='cuda', dim=0, **kwargs):
    if device == 'cuda': 
        model = model.cuda()
    
    model = MMDataParallel(model, dim=dim) 
    if kwargs.get('classes', None) is not None:
        model.CLASSES = kwargs['classes']
    model.cfg = cfg
    return model
    
    
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
        
        
    def train_step(self, *inputs):
        assert self.device_ids == [0], "this project is for only single gpu with ID == '0',\
                                        but device_ids is {self.device_ids}"
        # inputs[0]: data_batch, dict
        #    inputs[0].keys():  ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_masks']
        # inputs[1]: optimizer
        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        
       
        inputs = scatter_inputs(inputs, self.device_ids)       
        return self.module.train_step(*inputs[0])
    
    
    def forward(self, *inputs, **kwargs):
        """Override the original forward function.

        The main difference lies in the CPU inference where the data in
        :class:`DataContainers` will still be gathered.
        """
        # kwargs.keys = ['return_loss', 'rescale', 'img_metas', 'img']
       
        return super().forward(*inputs, **kwargs)
    

def build_detector(config, model_path, device='cuda:0', logger = None):
    from utils.checkpoint import load_checkpoint
    
    checkpoint = load_checkpoint(model_path, logger = logger)
    
    state_dict = checkpoint['state_dict']
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    meta = checkpoint['meta']
    optimizer = checkpoint['optimizer']
    
    if meta.get("model_cfg", None) is not None:
        model = build_model(meta['model_cfg'])
    else:
        model = build_model(config.model)
        
        
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    # load state_dict
    load_state_dict(model, state_dict, logger)
    
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model



def load_state_dict(module: torch.nn.Module, state_dict, logger = None):
    """
    Copies parameters and buffers from state_dict into module
    """
    
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    
    
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        
        # method of nn.Module
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    # break load->load reference cycle
    load = None  # type: ignore
    
    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]


    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0 :
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)  # type: ignore
        if logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

import warnings
from getpass import getuser
from socket import gethostname
import time
import os
import os.path as osp
from itertools import repeat
import collections.abc
from collections.abc import Mapping, Sequence
import torch
import  numpy as np

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')
    
    
def get_host_info():
    """Get hostname and username.

    Return empty string if exception raised, e.g. ``getpass.getuser()`` will
    lead to error in docker container
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    finally:
        return host
    
    

def is_tuple_of(seq, expected_type):

    if not isinstance(seq, tuple):
        return False
    
    for item in seq:
        if not isinstance(item, expected_type):
            return False

    return True

def is_list_of(seq, expected_type):
    """Check whether it is a sequence of list type.
    """
    if not isinstance(seq, list):
        return False
    
    for item in seq:
        if not isinstance(item, expected_type):
            return False

    return True


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

# multi GPU사용할때 
def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()

# simgle gpu: cuda,   multi gpu: mlu  
def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) == 1 else 'cpu'

# not use. 나중에 여유 될때 사용해서 training해보기
def auto_scale_lr(cfg, logger, num_gpus = 1):   
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): whole config.
        logger (logging.Logger): Logger.
    """
    
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return
    
    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return
    
    batch_size = cfg.data.train_dataloader.samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s). The total batch size is {batch_size}.')
    
    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')
        
        
def set_meta(cfg, args, env_info):
    meta = dict()
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['seed'] = cfg.seed
    meta['exp_name'] = os.path.basename(args.cfg) 
    return meta    



def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())



def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Default: True.

    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, str):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(
            item.lower() for item in suffix)

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def dict_to_pretty(cfg):
    if not isinstance(cfg, dict) : raise TypeError(f"cfg must be dict, but get {type(cfg)}")
    indent = 4

    def _indent(s_, num_spaces):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    def _format_basic_types(k, v, use_mapping=False):
        if isinstance(v, str):
            v_str = f"'{v}'"
        else:
            v_str = str(v)

        if use_mapping:
            k_str = f"'{k}'" if isinstance(k, str) else str(k)
            attr_str = f'{k_str}: {v_str}'
        else:
            attr_str = f'{str(k)}={v_str}'
        attr_str = _indent(attr_str, indent)

        return attr_str

    def _format_list(k, v, use_mapping=False):
        # check if all items in the list are dict
        if all(isinstance(_, dict) for _ in v):
            v_str = '[\n'
            v_str += '\n'.join(
                f'dict({_indent(_format_dict(v_), indent)}),'
                for v_ in v).rstrip(',')
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent) + ']'
        else:
            attr_str = _format_basic_types(k, v, use_mapping)
        return attr_str

    def _contain_invalid_identifier(dict_str):
        contain_invalid_identifier = False
        for key_name in dict_str:
            contain_invalid_identifier |= \
                (not str(key_name).isidentifier())
        return contain_invalid_identifier

    def _format_dict(input_dict, outest_level=False):
        r = ''
        s = []

        use_mapping = _contain_invalid_identifier(input_dict)
        if use_mapping:
            r += '{'
        for idx, (k, v) in enumerate(input_dict.items()):
            is_last = idx >= len(input_dict) - 1
            end = '' if outest_level or is_last else ','
            if isinstance(v, dict):
                v_str = '\n' + _format_dict(v)
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: dict({v_str}'
                else:
                    attr_str = f'{str(k)}=dict({v_str}'
                attr_str = _indent(attr_str, indent) + ')' + end
            elif isinstance(v, list):
                attr_str = _format_list(k, v, use_mapping) + end
            else:
                attr_str = _format_basic_types(k, v, use_mapping) + end

            s.append(attr_str)
        r += '\n'.join(s)
        if use_mapping:
            r += '}'
        return r

    text = _format_dict(cfg, outest_level=True)
    
    return text
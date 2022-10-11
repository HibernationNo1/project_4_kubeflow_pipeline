import torch
import os.path as osp
import re
import logging
from typing import Optional, Union, Sequence, Callable
from collections import OrderedDict


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: dict = {}
    
    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))
      
        
        
    @classmethod
    def register_scheme(cls,
                        prefixes: Union[str, Sequence[str]],
                        loader: Optional[Callable] = None,
                        force: bool = False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or Sequence[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register
    
    
    @classmethod
    def load_checkpoint(
            cls,
            filename: str,
            map_location: Optional[str] = None,
            logger: Optional[logging.Logger] = None
    ) -> Union[dict, OrderedDict]:
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Default: None
            logger (:mod:`logging.Logger`, optional): The logger for message.
                Default: None

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """
        checkpoint_loader = cls._get_checkpoint_loader(filename)      
        class_name = checkpoint_loader.__name__
        
        logger.info(f'load checkpoint from {class_name[10:]} path: {filename}')
        return checkpoint_loader(filename, map_location)
        


    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(
        filename: str,
        map_location: Optional[str] = None) -> Union[dict, OrderedDict]:
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('http://', 'https://'))
def load_from_http(
        filename: str,
        map_location: Optional[str] = None,
        model_dir: Optional[str] = None) -> Union[dict, OrderedDict]:
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (str, optional): directory in which to save the object,
            Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    from torch.utils.model_zoo import load_url
    checkpoint = load_url(
        filename, model_dir=model_dir, map_location=map_location)
    return checkpoint
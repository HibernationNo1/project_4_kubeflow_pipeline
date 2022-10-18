import copy
from abc import ABCMeta, abstractmethod
import os
from typing import Optional, List, Union, Dict
from collections import defaultdict
from typing import Iterable
import warnings
import numpy as np
import logging
from logging import FileHandler
import torch
import torch.nn as nn
from torch.optim import Optimizer

from utils.utils import get_time_str
from utils.log import create_logger, get_logger,LogBuffer
        
        

class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.
        각 layer에 대해 parameter initialization를 수행
        
    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. 
    Compared with ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg: Optional[dict] = None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""
        super().__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):    
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # initialization information을 기록할 dict선언
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True
            
            
            for name, param in self.named_parameters():
                # init_info : describes the initialization.
                self._params_init_info[param]['init_info'] = \
                                   f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                # tmp_mean_value : parameter가 수정되었는지 여부를 나타내는 값의 평균
                # 해당 값이 변경되었을 시 관련 initialization information를 update한다.
                self._params_init_info[param]['tmp_mean_value'] = param.data.mean()
          
            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        logger = create_logger(log_name = "initialization")

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:   # initializatiom대상 layer들만 init   
                logger.info(f'initialize {module_name} with init_cfg {self.init_cfg}')
                
                # 실질적으로 initialization을 수행하는 function
                initialize(self, self.init_cfg)

                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of the pre-trained model from being overwritten by the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for module in self.children():
                # update init infomation
                if hasattr(module, 'init_weights'):
                    module.init_weights()       # module별로 init_weights 수행 
                    # users may overload the `init_weights`
                    assert hasattr(module,'_params_init_info'), f'Can not find `_params_init_info` in {module}'
                    
                    init_info=f'Initialized by ' \
                        f'user-defined `init_weights`' \
                        f' in {module.__class__.__name__}'
                        
                    for name, param in module.named_parameters():
                        assert param in module._params_init_info, (
                            f'Find a new :obj:`Parameter` '
                            f'named `{name}` during executing the '
                            f'`init_weights` of '
                            f'`{module.__class__.__name__}`. '
                            f'Please do not add or '
                            f'replace parameters during executing '
                            f'the `init_weights`. ')

                        # The parameter has been changed during executing the
                        # `init_weights` of module
                        mean_value = param.data.mean()
                        if module._params_init_info[param]['tmp_mean_value'] != mean_value:
                            module._params_init_info[param]['init_info'] = init_info
                            module._params_init_info[param]['tmp_mean_value'] = mean_value
            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')
            
        if is_top_level_module:
            self._dump_init_info()

            for sub_module in self.modules():
                del sub_module._params_init_info

    # _dump_init_info는 single gpu인 경우에만 수행되어야 한다.
    def _dump_init_info(self):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.
        """

        logger = get_logger(name = "initialization")

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(f"{'-'*60}\n"\
                    'Name of parameter - Initialization information\n')
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f'\n{name} - {param.shape}: '
                        f"\n{self._params_init_info[param]['init_info']} \n")
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                logger.info(f'\n{name} - {param.shape}: '
                            f"\n{self._params_init_info[param]['init_info']} \n ")
                
        

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s  
 

class BaseInit:
    def __init__(self,
                 *,
                 bias: float = 0,
                 bias_prob: Optional[float] = None,
                 layer: Union[str, List, None] = None):
        self.wholemodule = False
        if not isinstance(bias, (int, float)):
            raise TypeError(f'bias must be a number, but got a {type(bias)}')

        if bias_prob is not None:
            if not isinstance(bias_prob, float):
                raise TypeError(f'bias_prob type must be float, \
                    but got {type(bias_prob)}')

        if layer is not None:
            if not isinstance(layer, (str, list)):
                raise TypeError(f'layer must be a str or a list of str, \
                    but got a {type(layer)}')
        else:
            layer = []

        if bias_prob is not None:
            self.bias = float(-np.log((1 - bias_prob) / bias_prob)) 
        else:
            self.bias = bias
        self.layer = [layer] if isinstance(layer, str) else layer

    def _get_init_info(self):
        info = f'{self.__class__.__name__}, bias={self.bias}'
        return info    
    

    
    
def _initialize(module: nn.Module, cfg: Dict, wholemodule: bool = False):
    # 특정 type의 initialization을 실행 
    # TODO : initialization algorithm을 추가해서 type의 개수가 늘어나면 registry로 관리
    # TODO : cfg.type == Pretrained  도 추가하기
    from initialization import NormalInit, XavierInit
    
    cp_cfg = copy.deepcopy(cfg)
    init_type = cp_cfg.pop('type')
    if init_type =='Normal':
        func = NormalInit(**cp_cfg)
    elif init_type =='Xavier':
        func = XavierInit(**cp_cfg)
    
    
    # wholemodule : override(재정의) 할 때 사용
    func.wholemodule = wholemodule
    func(module)



def _initialize_override(module: nn.Module, override: Union[Dict, List],
                         cfg: Dict) -> None:
    if not isinstance(override, (dict, list)):
        raise TypeError(f'override must be a dict or a list of dict, \
                but got {type(override)}')

    override = [override] if isinstance(override, dict) else override

    for override_ in override:

        cp_override = copy.deepcopy(override_)
        name = cp_override.pop('name', None)
        if name is None:
            raise ValueError('`override` must contain the key "name",'
                             f'but got {cp_override}')
        # if override only has name key, it means use args in init_cfg
        if not cp_override:
            cp_override.update(cfg)
        # if override has name key and other args except type key, it will
        # raise error
        elif 'type' not in cp_override.keys():
            raise ValueError(
                f'`override` need "type" key, but got {cp_override}')
        if hasattr(module, name):
            _initialize(getattr(module, name), cp_override, wholemodule=True)
        else:
            raise RuntimeError(f'module did not have attribute {name}, '
                               f'but init_cfg is {cp_override}.')



    
def initialize(module: nn.Module, init_cfg: Union[Dict, List[dict]]):
    """Initialize a module.

    Args:
        module (``torch.nn.Module``): the module will be initialized.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. OpenMMLab has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, and ``Pretrained``.

    Example:
            # 1개의 layer에 대한 initialization
        >>> module = nn.Linear(2, 3, bias=True)
        >>> init_cfg = dict(type='Constant', layer='Linear', val =1 , bias =2)
        >>> initialize(module, init_cfg)
        
            # n개의 layer에 대한 initialization
        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
        >>> # define key ``'layer'`` for initializing layer with different
        >>> # configuration
        >>> init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
                dict(type='Constant', layer='Linear', val=2)]
        >>> initialize(module, init_cfg)

            
           # define key``'override'`` to initialize some specific part in module
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg = dict(type='Constant', val=1, bias=2, layer='Conv2d',
        >>>                 override=dict(type='Constant', name='reg', val=3, bias=4))  
        >>> initialize(model, init_cfg)

        >>> model = ResNet(depth=50)
        >>> # Initialize weights with the pretrained model.
        >>> init_cfg = dict(type='Pretrained',
                checkpoint='torchvision://resnet50')
        >>> initialize(model, init_cfg)

        >>> # Initialize weights of a sub-module with the specific part of
        >>> # a pretrained model by using "prefix".
        >>> url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
        >>>     'retinanet_r50_fpn_1x_coco/'\
        >>>     'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        >>> init_cfg = dict(type='Pretrained',
                checkpoint=url, prefix='backbone.')
    """
    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(f'init_cfg must be a dict or a list of dict, \
                but got {type(init_cfg)}')

    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        cp_cfg = copy.deepcopy(cfg)
        override = cp_cfg.pop('override', None)
        
        _initialize(module, cp_cfg)
        
        if override is not None:
            cp_cfg.pop('layer', None)
            _initialize_override(module, override, cp_cfg)
        else:
            # All attributes in module have same initialization.
            pass



        

class ModuleList(BaseModule, nn.ModuleList):
    """ ModuleList in openmmlab.
        layer를 class내에서 정의 후 list()로 감싸면 해당 class의 instance는 layer를 반환하지 않는다.
        하지만 ModuleList를 통해 layer를 감싸면 해당 class의 instance는 layer를 반환하게 되며 
        instance별로 layer list를 관리할 수 있다.
    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 modules: Optional[Iterable] = None,
                 init_cfg: Optional[dict] = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)
        
        
        
class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``

    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
        max_epochs (int, optional): Total training epochs.
        max_iters (int, optional): Total training iterations.
    """
    def __init__(self,
                 model,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None,
                 **kwargs):
       
        
        assert hasattr(model, 'train_step')
        
        
        # check the type of `optimizer`
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, '
                        f'but optimizer["{name}"] is a {type(optim)}')
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f'optimizer must be a torch.optim.Optimizer object '
                f'or dict or None, but got {type(optimizer)}')
            
        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be a logging.Logger object, '
                            f'but got {type(logger)}')

        # check the type of `meta`
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(
                f'meta must be a dict or None, but got {type(meta)}')
        
        self.batch_size = kwargs.get('batch_size', None)
         
        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta
        self.work_dir = work_dir
        if work_dir is None: raise TypeError(f"work_dir must be specific, but work_dir is 'None'") 
        if not os.path.isdir(work_dir): os.makedirs(work_dir, exist_ok=True)
        
        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__
        
        self._rank, self._world_size = 0, 1
        self.timestamp = get_time_str()
        self.mode = None
        self._hooks = []    # hooks을 저장할 list
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        
        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')
        if max_epochs is not None:    self.train_unit_type = 'epoch'
        else:                               self.train_unit_type = 'iter'
        self._max_epochs = max_epochs
        self._max_iters = max_iters
  
        
        self.log_buffer = LogBuffer()
        
    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @abstractmethod # 상속받을 class에서 사용 
    def train(self):
        pass
    
    @abstractmethod
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl,
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        pass
    # TODO
    
    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr
        
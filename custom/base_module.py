import copy
from abc import ABCMeta
from re import M
from typing import Optional, List, Union, Dict
from collections import defaultdict
from typing import Iterable
import warnings
import numpy as np
from logging import FileHandler
import torch.nn as nn


from utils import get_logger
        
        

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
     
        # logger를 가져온다. 
        # TODO : initialized logger를 따로 선언하기
        logger = get_logger()

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                logger.info(f'initialize {module_name} with init_cfg {self.init_cfg}')
                
                # 실질적으로 initialization을 수행하는 function
                initialize(self, self.init_cfg)
 
                # prevent the parameters of the pre-trained model from being overwritten by the `init_weights`
                if self.init_cfg['type'] == 'Pretrained':
                    return

            for module in self.children():
                
                # update init infomation
                if hasattr(module, 'init_weights'):
                    module.init_weights()
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

        logger = get_logger()

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(
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
    
    
class NormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        mean (int | float):the mean of the normal distribution. Defaults to 0.
        std (int | float): the standard deviation of the normal distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.

    """

    def __init__(self, mean: float = 0, std: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def __call__(self, module: nn.Module) -> None:

        def init(m):
            if self.wholemodule:
                if hasattr(M, 'weight') and m.weight is not None:   nn.init.normal_(m.weight, self.mean, self.std)
                if hasattr(m, 'bias') and m.bias is not None:       nn.init.constant_(m.bias, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = [b.__name__ for b in m.__class__.__bases__]
                if len(set(self.layer) & set([layername] + basesname)):
                    if hasattr(M, 'weight') and m.weight is not None:   nn.init.normal_(m.weight, self.mean, self.std)
                    if hasattr(m, 'bias') and m.bias is not None:       nn.init.constant_(m.bias, self.bias)
           

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: mean={self.mean},' \
               f' std={self.std}, bias={self.bias}'
        return info
    
    
    
def update_init_info(module: nn.Module, init_info: str):
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.
    """
    assert hasattr(
        module,
        '_params_init_info'), f'Can not find `_params_init_info` in {module}'
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
            
            
class XavierInit(BaseInit):
    """Initialize module parameters with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks - Glorot, X. & Bengio, Y. (2010).
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

    Args:
        gain (int | float): an optional scaling factor. Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'``
            or ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self,
                 gain: float = 1,
                 distribution: str = 'normal',
                 **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module: nn.Module) -> None:

        def init(m):
            if self.wholemodule:
                assert self.distribution in ['uniform', 'normal']
                if hasattr(m, 'weight') and m.weight is not None:
                    if self.distribution == 'uniform': nn.init.xavier_uniform_(m.weight, gain=self.gain)
                    else:                              nn.init.xavier_normal_(m.weight, gain=self.gain)
                
                if hasattr(m, 'bias') and m.bias is not None:   nn.init.constant_(m.bias, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = [b.__name__ for b in m.__class__.__bases__]
                if len(set(self.layer) & set([layername] + basesname)):
                    assert self.distribution in ['uniform', 'normal']
                    if hasattr(m, 'weight') and m.weight is not None:
                        if self.distribution == 'uniform': nn.init.xavier_uniform_(m.weight, gain=self.gain)
                        else:                              nn.init.xavier_normal_(m.weight, gain=self.gain)
                        
                    if hasattr(m, 'bias') and m.bias is not None:   nn.init.constant_(m.bias, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: gain={self.gain}, ' \
               f'distribution={self.distribution}, bias={self.bias}'
        return info
    
    
def _initialize(module: nn.Module, cfg: Dict, wholemodule: bool = False):
    # 특정 type의 initialization을 실행 
    # TODO : initialization algorithm을 추가해서 type의 개수가 늘어나면 registry로 관리
    # TODO : cfg.type == Pretrained  도 추가하기
    if cfg.type =='Normal':
        func = NormalInit(cfg.distribution)
    elif cfg.type =='Xavier':
        func = XavierInit(cfg.distribution)
        
    
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





    
def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        

def update_init_info(module: nn.Module, init_info: str):
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.
    """
    assert hasattr(
        module,
        '_params_init_info'), f'Can not find `_params_init_info` in {module}'
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
            
            
class XavierInit(BaseInit):
    """Initialize module parameters with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks - Glorot, X. & Bengio, Y. (2010).
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

    Args:
        gain (int | float): an optional scaling factor. Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'``
            or ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self,
                 gain: float = 1,
                 distribution: str = 'normal',
                 **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module: nn.Module) -> None:

        def init(m):
            if self.wholemodule:
                xavier_init(m, self.gain, self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = [b.__name__ for b in m.__class__.__bases__]

                if len(set(self.layer) & set([layername] + basesname)):
                    xavier_init(m, self.gain, self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: gain={self.gain}, ' \
               f'distribution={self.distribution}, bias={self.bias}'
        return info
    

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
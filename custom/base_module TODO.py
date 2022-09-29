import copy
from abc import ABCMeta
from typing import Optional, List, Union
from collections import defaultdict
import numpy as np

import torch.nn as nn
from log import logger_initialized, log_recorder

class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

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

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter, which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters is initialized.
            
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`
            # When detecting the `tmp_mean_value` of the corresponding parameter is changed, update related initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]['init_info'] = \
                                   f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param]['tmp_mean_value'] = param.data.mean()
          
            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info
     
        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
       
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "default"
        logger = log_recorder[logger_name]

    
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                logger.info(f'initialize {module_name} with init_cfg {self.init_cfg}', logger=logger_name)
          
                if not isinstance(self.init_cfg, dict):
                    raise TypeError(f'init_cfg must be a dict , but got {type(self.init_cfg)}')

                cp_cfg = copy.deepcopy(self.init_cfg)
                override = cp_cfg.pop('override', None)
        
                # Initialize module parameters
                func = XavierInit(**self.init_cfg)
                func.wholemodule = wholemodule
                func(self)
                
                
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
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name: str):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

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
                print_log(
                    f'\n{name} - {param.shape}: '
                    f"\n{self._params_init_info[param]['init_info']} \n ",
                    logger=logger_name)

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
    
    
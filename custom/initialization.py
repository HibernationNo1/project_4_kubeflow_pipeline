import torch.nn as nn
from re import M

from base_module import BaseInit


def kaiming_init(module: nn.Module,     
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
        
def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
        
        
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

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: mean={self.mean},' \
               f' std={self.std}, bias={self.bias}'
        return info
    
    
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
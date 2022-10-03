
from itertools import repeat
import collections.abc
from log import logger_initialized, log_recorder

# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

def get_logger(name = None):
    if name is None:    # get logger at index 0
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "default"
    else:
        logger_name = name
    
    logger = log_recorder[logger_name]
    return logger 
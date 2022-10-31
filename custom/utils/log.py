# Copyright (c) OpenMMLab. All rights reserved.
import logging
from collections import defaultdict
from re import I
import sys
import subprocess
import cv2
import os.path as osp
import torch
import time
import os
from utils.utils import get_time_str

logger_initialized: dict = {}       # 어디서든 호출할 수 있도록 global선언
log_recorder: dict = {}         
log_info: dict = {}
# Example
# >>> from log import log_recorder
# >>>    logger = log_recorder[f'{log_name}']
# >>>    logger.info(f'key: \n{value}')     # log에 저장

TORCH_VERSION = torch.__version__

def set_logger_info(path, log_level):
    timestamp = get_time_str()
    log_info['timestamp'] = timestamp
    
    result_dir = osp.join(path, timestamp)
    log_info['result_dir'] = result_dir
    
    log_info['log_level'] = log_level
    os.makedirs(result_dir, exist_ok= True)
    return timestamp

def create_logger(log_name = None):
    if log_name is None: 
        return get_logger()
    elif log_name in list(logger_initialized.keys()):
        return get_logger(name=log_name)
    log_file = os.path.join(log_info['result_dir'], f'{log_name}.log')
    get_logger(name=log_name, log_file=log_file, log_level=log_info['log_level'])        
    logger = log_recorder[log_name]
    return logger
    

def is_rocm_pytorch():
    is_rocm = False
    if TORCH_VERSION != 'parrots':
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            is_rocm = True if ((torch.version.hip is not None) and
                               (ROCM_HOME is not None)) else False
        except ImportError:
            pass
    return is_rocm  # bool



def collect_env():
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available
    
    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name
    
        assert TORCH_VERSION != 'parrots'

        if is_rocm_pytorch():
            from torch.utils.cpp_extension import ROCM_HOME
            CUDA_HOME = ROCM_HOME
        else:
            from torch.utils.cpp_extension import CUDA_HOME
            
        env_info['CUDA_HOME'] = CUDA_HOME
        
        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
                release = nvcc.rfind('Cuda compilation tools')
                build = nvcc.rfind('Build ')
                nvcc = nvcc[release:build].strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc
            
    try:
        # Check C++ Compiler.
        # For Unix-like, sysconfig has 'CC' variable like 'gcc -pthread ...',
        # indicating the compiler used, we use this to get the compiler name
        import sysconfig
        cc = sysconfig.get_config_var('CC')
        if cc:
            cc = osp.basename(cc.split()[0])
            cc_info = subprocess.check_output(f'{cc} --version', shell=True)
            env_info['GCC'] = cc_info.decode('utf-8').partition(
                '\n')[0].strip()
        else:
            # on Windows, cl.exe is not in PATH. We need to find the path.
            # distutils.ccompiler.new_compiler() returns a msvccompiler
            # object and after initialization, path to cl.exe is found.
            import locale
            import os
            from distutils.ccompiler import new_compiler
            ccompiler = new_compiler()
            ccompiler.initialize()
            cc = subprocess.check_output(
                f'{ccompiler.cc}', stderr=subprocess.STDOUT, shell=True)
            encoding = os.device_encoding(
                sys.stdout.fileno()) or locale.getpreferredencoding()
            env_info['MSVC'] = cc.decode(encoding).partition('\n')[0].strip()
            env_info['GCC'] = 'n/a'
    except subprocess.CalledProcessError:
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass
    
    env_info['OpenCV'] = cv2.__version__
    
    return env_info         




def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:       
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]


    # only rank 0 will add a FileHandler
    if log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    logger_initialized[name] = True

    log_recorder[name] = logger
    
# simplefy
# def get_logger(name = None):
#     if name is None:    # get logger at index 0
#         logger_names = list(logger_initialized.keys())
#         logger_name = logger_names[0] if logger_names else "default"
#     else:
#         logger_name = name
    
#     logger = log_recorder[logger_name]
#     return logger
    


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:

            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')







import numpy as np


class LogBuffer:

    def __init__(self):
        self.val_history = dict()
        self.n_history = dict()
        self.output = dict()
        self.log_output = dict()
        self.ready = False

    def clear(self) -> None:
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self) -> None:
        self.output.clear()
        self.ready = False

    def update(self, vars: dict, count: int = 1) -> None:
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n: int = 0) -> None:
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True
        
    
    def log(self, n):
      
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.log_output[key] = avg
    
    def clear_log(self):
        self.log_output.clear()






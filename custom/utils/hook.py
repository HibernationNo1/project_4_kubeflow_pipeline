from typing import List, Optional, Union, Dict
import time
from collections import OrderedDict
import logging
import os
import os.path as osp

import torch
from torch.nn.utils import clip_grad

from utils.utils import is_tuple_of, is_list_of, dict_to_pretty
from base_module import BaseRunner
from mmdet_taeuk4958.models.dense_heads.rpn_head import RPNHead ####
# from models.maskrcnn.rpn import RPNHead

def is_method_overridden(method, base_class, derived_class):
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method

class Hook:
    stages = ('before_run', 'before_train_epoch', 'before_train_iter',
              'after_train_iter', 'after_train_epoch', 'before_val_epoch',
              'before_val_iter', 'after_val_iter', 'after_val_epoch',
              'after_run')

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_val_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_val_iter(self, runner):
        self.after_iter(runner)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)

    def is_last_epoch(self, runner):
        return runner.epoch + 1 == runner._max_epochs

    def is_last_iter(self, runner):
        return runner.iter + 1 == runner._max_iters

    def get_triggered_stages(self):
        """
            각각의 stage에 대해 활성화 할 hook을 select한다. 
            stage : before_run              학습 시작 전
                    before_train_epoch      epoch시작 전    
                    before_train_iter       iter시작 전
                    after_train_iter        1 iter이 끝난 후
                    after_train_epoch       1 epoch가 끝난 후 
                    before_val_epoch        
                    before_val_iter          
                    after_val_iter
                    after_val_epoch
                    after_run               학습 종료 후
        """
        trigger_stages = set()
        for stage in Hook.stages:
            if is_method_overridden(stage, Hook, self):
                trigger_stages.add(stage)


        
        # some methods will be triggered in multi stages
        # use this dict to map method to stages.
        method_stages_map = {
            'before_epoch': ['before_train_epoch', 'before_val_epoch'],
            'after_epoch': ['after_train_epoch', 'after_val_epoch'],
            'before_iter': ['before_train_iter', 'before_val_iter'],
            'after_iter': ['after_train_iter', 'after_val_iter'],
        }
        for method, map_stages in method_stages_map.items():
            if is_method_overridden(method, Hook, self):
                trigger_stages.update(map_stages)

        
        return [stage for stage in Hook.stages if stage in trigger_stages]

class NumClassCheckHook(Hook):
    
    def before_val_epoch(self, runner):
        """Check whether the dataset in val epoch is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)
        
    
    def before_train_epoch(self, runner):
        """Check whether the training dataset is compatible with head.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        self._check_head(runner)
        
        
    def _check_head(self, runner):
        """Check whether the `num_classes` in head matches the length of
        `CLASSES` in `dataset`.

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
      
        model = runner.model
        dataset = runner.data_loader.dataset
        
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set `CLASSES` '
                f'in the {dataset.__class__.__name__} and'
                f'check if it is consistent with the `num_classes` '
                f'of head')
        
        else:
            assert type(dataset.CLASSES) is not str, \
                (f'`CLASSES` in {dataset.__class__.__name__}'
                 f'should be a tuple of str.'
                 f'Add comma if number of classes is 1 as '
                 f'CLASSES = ({dataset.CLASSES},)')
                
            for name, module in model.named_modules():
                if hasattr(module, 'num_classes') and not isinstance(module, RPNHead):
                    assert module.num_classes == len(dataset.CLASSES), \
                        (f'The `num_classes` ({module.num_classes}) in '
                         f'{module.__class__.__name__} of '
                         f'{model.__class__.__name__} does not matches '
                         f'the length of `CLASSES` '
                         f'{len(dataset.CLASSES)}) in '
                         f'{dataset.__class__.__name__}')
                        

class LoggerHook(Hook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and
    saved in json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedRunner is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        interval_exp_name (int, optional): Logging interval for experiment
            name. This feature is to help users conveniently get the experiment
            information from screen or log file. Default: 1000.
        out_dir (str, optional): Logs are saved in ``runner.work_dir`` default.
            If ``out_dir`` is specified, logs will be copied to a new directory
            which is the concatenation of ``out_dir`` and the last level
            directory of ``runner.work_dir``. Default: None.
            `New in version 1.3.16.`
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be copied to ``out_dir``.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.3.16.`
        keep_local (bool, optional): Whether to keep local log when
            :attr:`out_dir` is specified. If False, the local log will be
            removed. Default: True.
            `New in version 1.3.16.`
    """

    def __init__(self,
                 max_epochs,
                 ev_iter,        # iters_per_epochs
                 by_epoch: bool = True,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[str] = None,
                 out_suffix: str = '.log',
                 log_file_name: str = 'RUN_log',
                 keep_local: bool = True):
        self.max_epochs = max_epochs 
        self.ev_iter = ev_iter
        self.iter_count = 1
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.by_epoch = by_epoch
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name
        
        self.out_dir = out_dir  
        
        if not (out_dir is None or isinstance(out_dir, str)
                or is_tuple_of(out_dir, str)):
            raise TypeError('out_dir should be  "None" or string or tuple of '
                            'string, but got {out_dir}')
        self.out_suffix = out_suffix
        self.log_file_name = log_file_name + out_suffix

        self.keep_local = keep_local
    
    
    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items
    
    def get_mode(self, runner) -> str:
        if runner.mode == 'train':
            if 'time' in runner.log_buffer.output:
                mode = 'train'
            else:
                mode = 'val'
        elif runner.mode == 'val':
            mode = 'val'
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return mode
    
    def get_epoch(self, runner) -> int:
        if runner.mode == 'train':
            epoch = runner.epoch + 1
        elif runner.mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch
    
    
    def _get_max_memory(self, runner) :
        """
            Size of tensor allocated to GPU (unit: MB)
        """
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        return mem_mb.item()
    
    def get_iter(self, runner, inner_iter: bool = False) -> int:
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter
    
    def _log_info(self, log_dict: Dict, runner) -> None:
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)  # type: ignore
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'  # type: ignore

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}]' \
                          f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f'{lr_str}, '

            import datetime
            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, ' \
                           f'data_time: {log_dict["data_time"]:.3f}, '
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        
        
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        
        log_str += ', '.join(log_items)
        
        runner.logger.info(log_str)

        
        
    def write_log(self, status, log_dict_list: list):    
        # dump log in .log format

        if not is_list_of(log_dict_list, dict): raise TypeError(f"element of list must be dict, but was not.")
        
        text_log = '\n'
        if status in ['before_run', 'after_run']:
            num_bar = 60
        elif status in ['before_epoch', 'after_epoch']:
            num_bar = 45
        elif status in ['before_iter', 'after_iter']:
            num_bar = 30
        
        for i in range(num_bar):
            if i == num_bar//2:
                text_log += f'< {status} >'
            text_log += '-'
            
        for log_dict in log_dict_list:
            text_log = text_log + "\n" + dict_to_pretty(log_dict) + "\n"
            
        if osp.isfile(self.log_file_path):
            with open(self.log_file_path, 'a+') as f:
                f.write(f"{text_log}")
        else:
            with open(self.log_file_path, 'w') as f:        
                f.write(f"{text_log}")
    
        
                
    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break
            
        if self.out_dir is not None:
            runner.logger.info(
                f'Text logs will be saved to {self.out_dir} after the training process.')
            self.log_file_path = osp.join(self.out_dir, self.log_file_name)
        else:
            self.log_file_path = osp.join(runner.work_dir, self.log_file_name)
              
        self.start_iter = runner.iter
        log_dict_list = []
        log_dict_list.append(dict(
            bach_size = runner.get('batch_size') ,
            max_iteration = runner.get('_max_iters'),
            max_epoche = runner.get('_max_epochs')
        ))
        
        if len(log_dict_list) != 0:
            self.write_log('before_run', log_dict_list)  
        
        self.r_t = time.time()  
            
        
        
    def before_train_epoch(self, runner):
        log_dict = dict(
            start_epoch = f'({self.get_epoch(runner)}/{self.max_epochs})',
            iterd_per_epochs = self.ev_iter
            )
        self.write_log('before_epoch', [log_dict])
        runner.log_buffer.clear()  # clear logs of last epoch
        self.e_t = time.time()      # epoch start time 
    
    def before_train_iter(self, runner):            
        log_dict_meta = dict(epoch=f'({self.get_epoch(runner)}/{self.max_epochs})',
                             iter =f'({self.get_iter(runner, inner_iter=True)}/{self.ev_iter})')
        self.write_log('before_iter', [log_dict_meta])
        self.i_t = time.time()      # iter start time 
                
        
    def after_train_iter(self, runner) -> None:   
        self.iter_count += 1
        log_mode = 'train'
        
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):   # training unit: epoch, iter +1
            log_mode = 'val'
            runner.log_buffer.average(self.interval)   
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):   # training unit: iter, iter +1
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:    # at last epoch
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()
        
        
        current_iter = self.get_iter(runner, inner_iter=True)
        time_spent_iter = round(time.time() - self.i_t, 2)
        remain_time = self.compute_remain_time(time_spent_iter)['remain_time']
        log_dict_meta = OrderedDict(
            time_spent_iter = time_spent_iter,
            mode=log_mode,
            epoch=f'({self.get_epoch(runner)}/{self.max_epochs})',
            iter=f'({current_iter}/{self.ev_iter})',
            remain_time = remain_time)
        
        runner.log_buffer.log(self.interval)

        log_dict_loss = dict(**runner.log_buffer.log_output) 
        del log_dict_loss['data_time'], log_dict_loss['time']
        self.write_log("after_iter", [log_dict_meta, log_dict_loss])
        runner.log_buffer.clear_log()
        
        if log_mode == 'train':
            print(f"Time remaining: [{remain_time}]\
                    epoch: [{self.get_epoch(runner)}/{self.max_epochs}]\
                    iter: [{self.get_iter(runner, inner_iter=True)}/{self.ev_iter}]")


    def after_train_epoch(self, runner) -> None:
        if runner.log_buffer.ready:
            self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()
     
        time_spent_epoch = time.time() - self.e_t
        log_dict_meta = OrderedDict(
            time_spent_epoch = self.compute_sec_to_h_d(time_spent_epoch),
            end_epoch=self.get_epoch(runner), 
            **self.compute_remain_time(time_spent_epoch))
        
        self.write_log("after_epoch", [log_dict_meta])

   
    def after_run(self, runner):
        log_dict_meta = OrderedDict(
            training_time = self.compute_sec_to_h_d(time.time() - self.r_t)
            )
        
        log_dict_loss = dict(**runner.log_buffer.log_output) 
        self.write_log("after_run", [log_dict_meta, log_dict_loss])
        runner.log_buffer.clear_log()
         
    
    def log(self, runner) :
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            iter=cur_iter)
        
        # learning rate 할당
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})
                
        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)
        
        log_dict = dict(log_dict, **runner.log_buffer.output) 
        self._log_info(log_dict, runner)
        
        
    def compute_sec_to_h_d(self, sec):
        if sec < 60: return f'00:00:{f"{int(sec).zfill(2)}"}'
        
        minute = sec//60
        if minute < 60: return f"00:{f'{int(minute)}'.zfill(2)}:{f'{int(sec%60)}'.zfill(2)}"
        
        hour = minute//60
        if hour < 24: return f"{f'{int(hour)}'.zfill(2)}:{f'{int(minute%60)}'.zfill(2)}:{f'{int(sec%3600)}'.zfill(2)}"
        
        day = hour//24
        return f"{day}day {f'{int(hour%24)}'.zfill(2)}:{f'{int(minute%(60*24))}'.zfill(2)}:{f'{int(sec%(3600*24))}'.zfill(2)}"
         
         
    def compute_remain_time(self, time_spent):
        time_dict = dict()
        
        max_iter = self.max_epochs*self.ev_iter      # total iters for training 
        remain_iter = max_iter - self.iter_count
        time_dict['remain_time'] = self.compute_sec_to_h_d(time_spent * remain_iter)           
        
        return time_dict           
    

class IterTimerHook(Hook):
    
    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        runner.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()


class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``.
            `Changed in version 1.3.16.`
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        save_last (bool, optional): Whether to force the last checkpoint to be
            saved regardless of interval. Default: True.
        sync_buffer (bool, optional): Whether to synchronize buffers in
            different gpus. Default: False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`

    .. warning::
        Before v1.3.16, the ``out_dir`` argument indicates the path where the
        checkpoint is stored. However, since v1.3.16, ``out_dir`` indicates the
        root directory and the final path to save checkpoint is the
        concatenation of ``out_dir`` and the last level directory of
        ``runner.work_dir``. Suppose the value of ``out_dir`` is "/path/of/A"
        and the value of ``runner.work_dir`` is "/path/of/B", then the final
        path will be "/path/of/A/B".
    """

    def __init__(self,
                 interval=-1,           # Target unit of epoch(or iter) performing CheckpointHook
                 by_epoch=True,         # True : run CheckpointHook unit by epoch , False : unit by iter
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,     # number of model which save by .pth format  
                                        # why?: when save model whevery epoch(or iter), the number increases.
                                        # so, delete number from 'last epoch' to 'max_keep_ckpts'
                 save_last=True,
                 file_client_args=None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch    
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir      # runner의 work_dir로 대체하기 위해 선언 필요
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.file_client_args = file_client_args
        
        
        
    def before_run(self, runner):   
        if not self.out_dir:
            self.out_dir = runner.work_dir
        
        runner.logger.info(f'Checkpoints will be saved to {self.out_dir} ')
        
        # self.args['create_symlink'] = True
        
    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

       
        # run whenever epoch is a multiple of self.interval(default: 1)
        if self.every_n_epochs(runner, self.interval) \
                           or (self.save_last and self.is_last_epoch(runner)):
                               
            runner.logger.info(f'Saving checkpoint at {runner.epoch + 1} epochs')
            self._save_checkpoint(runner)
            
            
    def _save_checkpoint(self, runner):
        """Save the current checkpoint and delete unwanted checkpoint."""
        # save meta, parameters of model, optimazers 
        runner.save_checkpoint(self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        
        if runner.meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get('filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)

            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = osp.join(self.out_dir, cur_ckpt_filename)

       
        # remove other checkpoints      # do not 
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            filename_tmpl = self.args.get('filename_tmpl', name)
            
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            
            for _step in redundant_ckpts: 
                ckpt_path = osp.join(self.out_dir, filename_tmpl.format(_step))
                if osp.isfile(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break
    
    
    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(f'Saving checkpoint at {runner.iter + 1} iterations')
        
            self._save_checkpoint(runner)
                

class OptimizerHook(Hook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """
    def __init__(self, grad_clip=None, detect_anomalous_params=False):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    
    def after_train_iter(self, runner):
        """
            최적화 수행
        """
        # initialize gradient
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
            
        # Computes the gradient of current tensor 
        runner.outputs['loss'].backward()

        # optimize (back propagation)
        runner.optimizer.step()    
        
    
    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)
        
        
    def detect_anomalous_parameters(self, loss, runner):
        """
            학습에 사용되지 않는 model의 parameter를 찾을 수 있다.
        """
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()
        
        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')
    
class StepLrUpdaterHook(Hook):
    
    def __init__(self,
                 step: Union[int, List[int]],
                 gamma: float = 0.1,            # learning rate에 변화를 줄 때 사용되는 상수
                 min_lr: Optional[float] = None,
                 by_epoch: bool = True,
                 warmup: Optional[str] = None,
                 warmup_iters: int = 0,
                 warmup_ratio: float = 0.1,
                 warmup_by_epoch: bool = False  # TODO : warmup의 개념 사용해보기
                 ) -> None:
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma      
        self.min_lr = min_lr
        
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'
        
        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters: Optional[int] = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch
        
        # TODO : warmup의 개념 사용해보기
        if self.warmup_by_epoch:
            self.warmup_epochs: Optional[int] = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None
            
        self.base_lr: Union[list, dict] = []  # initial lr for all param groups
        self.regular_lr: list = []  # expected lr if no warming up is performed
        

    def get_lr(self, runner: 'BaseRunner', base_lr: float):
        # learning rate에 특정 값을 복하여 변화를 준다
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr
    
    
    def get_regular_lr(self, runner: 'BaseRunner'):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]
        
        
    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                    lr_groups):
                param_group['lr'] = lr
    
    

    def before_run(self, runner: 'BaseRunner'):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params

        # 각 group에 적용할 learning rate
        for group in runner.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [group['initial_lr'] for group in runner.optimizer.param_groups]
        
    
    def before_train_epoch(self, runner: 'BaseRunner'):
        if self.warmup_iters is None:
            epoch_len = len(runner.data_loader)  # type: ignore
            self.warmup_iters = self.warmup_epochs * epoch_len  # type: ignore

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)   # 변동된(또는 되지 않은) learning rate
        self._set_lr(runner, self.regular_lr)           # 각 parameter group에 대해 learning rete적용
    
    
    def before_train_iter(self, runner: 'BaseRunner'):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
    
    
    def get_warmup_lr(self, cur_iters: int):
        # learning rate에 warmup type에 따른 계산식을 통해 특정 값을 적용
        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)
import torch
import time
import os.path as osp
from base_module import BaseRunner
from utils.utils import is_list_of, get_host_info

from utils.hook import (Hook,
                        StepLrUpdaterHook, 
                        OptimizerHook, 
                        CheckpointHook, 
                        IterTimerHook, 
                        TextLoggerHook, 
                        NumClassCheckHook)
from utils.checkpoint import save_checkpoint as sc_save_checkpoint 
priority_dict = {'HIGHEST' : 0,
                 'VERY_HIGH' : 10,
                 'HIGH' : 30,
                 'ABOVE_NORMAL' : 40,
                 'NORMAL' : 50,
                 'BELOW_NORMAL' : 60,
                 'LOW' : 70,
                 'VERY_LOW' : 90,
                 'LOWEST' : 100}

        
class EpochBasedRunner(BaseRunner):
    def run(self, data_loader, flow, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        
        
        mode, iter = flow
        if not isinstance(mode, str): 
            raise TypeError(f'mode in workflow must be a str, but got {type(mode)}') 
        if not isinstance(iter, int) : 
            raise TypeError(f'{self.train_unit_type} in workflow must be a int, but got {type(iter)}') 
                        
        work_dir = self.work_dir
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        if self._max_epochs is not None:        
            self.logger.info(f'mode: {mode}, max: {self._max_epochs} epochs')
            self._max_iters = self._max_epochs * len(data_loader)            # user가 설정한 epoch횟수에 따라 예상되는 iter값
        else:
            self.logger.info(f'mode: {mode}, max: {self._max_iters} iters')
            self._max_epochs = round(self._max_iters / len(data_loader), 2)  # user가 설정한 iter횟수에 따라 예상되는 epoch값 
        self.iterd_per_epochs = len(data_loader)
        work_dir = self.work_dir
        
        if not hasattr(self, mode):
            raise ValueError(f'runner has no method named "{mode}" \
                               to run an epoch')
        
        self.call_hook('before_run')
        if self._max_epochs is not None:        # epoch단위로 학습
            
            while self.epoch < self._max_epochs:
                
                epoch_runner = getattr(self, mode)      # mode에 해당되는 method 호출
                
                for _ in range(self._max_epochs):
                    epoch_runner(mode, data_loader, **kwargs)
                    
                
        else:                                   # iter단위로 학습
            
            while self.iter < self._max_iters:
                pass
                
            pass
            
            
                
    def train(self, mode, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook('before_train_iter')
            exit()
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1
            if self.train_unit_type == "iter" and self.iter == self._max_iters - 1: break
            
        self.call_hook('after_train_epoch')
        self._epoch += 1
 
            
        
    
    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called
                "before_run"
                "before_train_epoch"
                "before_train_iter"
                "after_train_iter"
                "after_train_epoch"
                "before_val_epoch"
                "before_val_iter"
                "after_val_iter"
                "after_run"
                
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
            
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

    def current_momentum(self):
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums
    
    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')

        # 우선 순위 설정
        for key in list(priority_dict.keys()):
            if priority == key : 
                priority = priority_dict[key]
                priority_dict[f'{priority}'] = key
                hook.priority = priority
                break
   
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)
            
    
    def load_hook(self, config, hook_type: str):
        # hook을 반드시 사용하는 것으로 설계
        
        if hook_type == "custom_hooks":
             # custom hook은 여러개 사용 가능
            if not isinstance(config, list):
                config = [config]
            
            for i, item in enumerate(config):
                if not isinstance(item, dict): raise TypeError(f"{hook_type}_config[{i}] type must be dict,\
                                                               but type is {type(item)}")
                hook_cfg = item.copy()
                priority = hook_cfg.pop('priority', 'NORMAL')
                custom_hook_type = hook_cfg.pop('type', 'NumClassCheckHook')

                if custom_hook_type == 'NumClassCheckHook':
                    hook = NumClassCheckHook()
                self.register_hook(hook, priority=priority)
            return
           
            
        if not isinstance(config, dict) : raise TypeError(f"{hook_type}_config type must be dict, but {hook_type}_config type is {type(config)}")
        
        if hook_type=="lr": 
            hook = StepLrUpdaterHook(**config) 
            priority = 'VERY_HIGH'
            
        elif hook_type=="optimizer": 
            hook = OptimizerHook(**config) 
            priority='ABOVE_NORMAL'
            
        elif hook_type=="checkpoint": 
            hook = CheckpointHook(**config) 
            priority='NORMAL'
        
        elif hook_type=="timer": 
            # config : no contents
            hook = IterTimerHook(**config) 
            priority='LOW'
            
        elif hook_type=="log": 
            # log hook은 여러개 사용 가능
            priority='VERY_LOW'
            text_cfg = dict(interval = config['interval'],
                            out_dir = self.work_dir,
                            out_suffix = '.log')
            
            for info in config['hooks']:
                if info['type'] == 'TensorboardLoggerHook': 
                    pass        # TODO
                elif info['type'] == 'TextLoggerHook': 
                    hook = TextLoggerHook(**text_cfg)
                self.register_hook(hook, priority=priority)
            return
            
        
        self.register_hook(hook, priority=priority)
        
    
    # TODO : hook사용 이유
    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                timer_config=dict(),
                                custom_hooks_config=None):
        
        self.load_hook(lr_config, "lr")
        self.load_hook(optimizer_config, "optimizer")
        self.load_hook(checkpoint_config, "checkpoint")
        self.load_hook(timer_config, "timer")
        self.load_hook(log_config, "log")
        self.load_hook(custom_hooks_config, "custom_hooks")
      
    def get_hook_info(self):
        # Get hooks info in each stage
        stage_hook_map = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = priority_dict[f'{hook.priority}']
            except ValueError:
                priority = hook.priority
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

 
        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)  
    
    
    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        
        if self.meta is not None:
            # 같은 값을 가진 meta를 기대하지만, 아닐 경우
            meta.update(self.meta)

        meta.update(epoch=self.epoch + 1, iter=self.iter)
        
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        sc_save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
    
    
    def get(self, att_name: str):
        try:
            return getattr(self, att_name)
        except:
            return None
            # raise AttributeError(f"{self.__class__.__name__} object has no attribute {att_name}")
       
        
       
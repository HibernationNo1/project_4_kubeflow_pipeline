from abc import ABCMeta
import numpy as np

from torch.utils.data import Sampler

class GroupSampler(Sampler):
    def __init__(self, dataset, batch_size=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset      
        self.batch_size = batch_size
        
        # width, height비교 flag
        self.flag = dataset.flag.astype(np.int64)       # [0 or 1, 0 or 1, ... 0 or 1]  0 : width > height, 1 : width < height
        self.group_sizes = np.bincount(self.flag)       # [count of 0, count of 1]
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(size / self.batch_size)) * self.batch_size
    # TODO
            

class RandomSampler(metaclass=ABCMeta):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """
    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self
        self.rng = np.random.mtrand._rand
    # TODO
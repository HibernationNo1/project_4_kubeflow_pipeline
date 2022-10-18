from abc import ABCMeta
import numpy as np

from torch.utils.data import Sampler

class GroupSampler(Sampler):
    def __init__(self, dataset, batch_size=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset      
        self.batch_size = batch_size
        
        # image의 width, height비교 flag
        self.flag = dataset.flag.astype(np.int64)       # [0 or 1, 0 or 1, ... 0 or 1]  0 : width > height, 1 : width < height
        self.group_sizes = np.bincount(self.flag)       # [count of 0, count of 1]
        self.num_samples = 0
        for size in self.group_sizes:
            self.num_samples += int(np.ceil(size / self.batch_size)) * self.batch_size
    
    def __iter__(self):
        indices = []        # 각 iamge의 index를 담을 list
        
        for i, size in enumerate(self.group_sizes):
            # image를 width, height 기준으로 나뉘어진 group
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]        
            assert len(indice) == size
            np.random.shuffle(indice)   # shuffle 한 번 해주고
            num_extra = int(np.ceil(size / self.batch_size)    # batch size에 대한 나머지 숫자
                            ) * self.batch_size - len(indice)
            indice = np.concatenate(                                # 나머지 숫자만큼의 image를 rendom하게 뽑은 후
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)                                  # batch size에 딱 맞는 len을 갖도록 append
        
        indices = np.concatenate(indices)
        indices = [     # batch_size의 길이를 가진 list의 list
            indices[i * self.batch_size:(i + 1) * self.batch_size]
            for i in np.random.permutation(
                range(len(indices) // self.batch_size))
        ]
        # len(indices) == round(number of image / batch_size, 1)
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()         # 다시 1 dimention의 list로 

        assert len(indices) == self.num_samples
        return iter(indices)
            
    
    def __len__(self):
        return self.num_samples
            

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
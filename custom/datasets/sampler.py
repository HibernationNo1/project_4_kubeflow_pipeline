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
    

def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)
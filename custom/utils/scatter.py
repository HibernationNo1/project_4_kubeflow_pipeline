from typing import List, Union
import torch
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel._functions import Scatter as OrigScatter
from datasets.datacontainer import DataContainer

def synchronize_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]],
                                   [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()       # get current stream
                main_stream.wait_stream(streams[0])             # Synchronizes with another stream.
                output.record_stream(main_stream)               # Ensures that the tensor memory is not reused
                                                                # for another tensor until all current work 
                                                                # queued on stream are complete.
    else:
        raise Exception(f'Unknown type {type(output)}.')
    



    
    
def forward(target_gpus, input):
    input_device = get_input_device(input)
    streams = None
    # input_device == -1면 CPU에 data가 할당되어 있는 경우
    if input_device == -1 and target_gpus != [-1]:  
        # Perform CPU to GPU copies in a background stream
        streams = [_get_stream(device) for device in target_gpus]

    outputs = scatter(input, target_gpus, streams)
    # Synchronize with the copy stream
    if streams is not None:
        synchronize_stream(outputs, target_gpus, streams)

    return tuple(outputs) if isinstance(outputs, list) else (outputs, )

    
    



def scatter(input, devices, streams=None):
    """Scatters tensor across single GPU."""
    
    if streams is None:
        streams = [None] * len(devices)
    
   
    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1       # chunk_size == 1 if using single GPU
        outputs = [
            scatter(input[i], [devices[i // chunk_size]],
                    [streams[i // chunk_size]]) for i in range(len(input))
        ]
        return outputs

    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        stream = streams[0] if output.numel() > 0 else None
    
        if devices != [-1]:
            # tensor를 gpu에 할당
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=True)

        return output
    else:
        raise Exception(f'Unknown type {type(input)}.')

    
def get_input_device(input):
    """
        input이 gpu에 할당이 안되어있다면(if input.is_cuda is False) return -1
        
        expected: return -1
    """
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception(f'Unknown type {type(input)}.')
    



    
# original: class Scatter
def forward_scatter(target_gpus, input):
    input_device = get_input_device(input)

    streams = None
    # input_device == -1면 아직 CPU에 data가 할당되어 있는 경우
    if input_device == -1 and target_gpus != [-1]:  
        # Perform CPU to GPU copies in a background stream
        streams = [_get_stream(device) for device in target_gpus]
    
    outputs = scatter(input, target_gpus, streams)
    # Synchronize with the copy stream
    if streams is not None:
        synchronize_stream(outputs, target_gpus, streams)
    
    return tuple(outputs) if isinstance(outputs, list) else (outputs, )   
    
    

# 단일 GPU를 목적으로 설계했기 때문에 기존 scatter와 달라진 점 거의 없음
# org : parallel > scatter_gather.py > scatter
def parallel_scatter(inputs, target_gpus = [0], dim = 0):
    """Scatter inputs to target gpus.
    """
  
    def scatter_map(obj):
        """
        Args:
            obj (_type_) is in ['tuple', 'dict', 'tuple', 'DataContainer']
        """
        
        # simplify code
        # batch_list = []
        # if isinstance(obj, tuple):
        #     for obj_ in obj:
        #         if isinstance(obj_, dict):
        #             outs = dict()
                    
        #             for key, value in obj_.items():
        #                 if value.cpu_only:
        #                     out= value.data
        #                 else:
        #                     out= forward_scatter(target_gpus, value.data)
        #                 outs[key] = out
        #         else:
        #             outs = obj_
                    
        #         batch_list.append(outs)
        # return [tuple(batch_list)]
                    
        if isinstance(obj, torch.Tensor):
            assert target_gpus != [-1], "Use only GPU, not CPU"
            return OrigScatter.apply(target_gpus, None, dim, obj)
        
        if isinstance(obj, DataContainer): 
            if obj.cpu_only:
                return obj.data
            else:
                return forward_scatter(target_gpus, obj.data)
        
        if isinstance(obj, tuple) and len(obj) > 0:      
            out = list(zip(*map(scatter_map, obj)))    
            return out
            
        
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        
        if isinstance(obj, dict) and len(obj) > 0:     
            out = list(map(dict, zip(*map(scatter_map, obj.items()))))
            return out
        
        return [obj for targets in target_gpus]
        
        
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None
            
# org : scatter_kwargs            
def scatter_inputs(inputs, target_gpus = [0], dim=0):
        """Scatter with support for kwargs dictionary."""
        # inputs data를 각 gpu에 나누어 할당하고, .contiguous() .cuda() 를 적용
        
        # type(inputs): tuple,       len(inputs) = 2
        # type(inputs[0]): dict,     ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_masks'] 
        #      img.shape = (batch_size, channel, height, width)
        #      else, len(key) == batch_size
        # type(inputs[0]): optimizer
        cattered_inputs = parallel_scatter(inputs, target_gpus, dim) if inputs else [] 
        # list로 한 차원 더 씌워지는 것 말고 구조변화는 없음
        # type(cattered_inputs): list,          len(cattered_inputs): 1
        # type(cattered_inputs[0]): tuple ,      len(cattered_inputs[0]): 2   
        #    type(cattered_inputs[0][0]): dict,     
        #       cattered_inputs[0][0].keys(): ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_masks']
        #    type(cattered_inputs[0][1]): optimizer,
       

        inputs = tuple(cattered_inputs)
        return inputs


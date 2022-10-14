import torch
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel._functions import Scatter as OrigScatter

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
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception(f'Unknown type {type(output)}.')
    
    
def scatter(input, devices, streams=None):
    """Scatters tensor across multiple GPUs.
        // 해당 project는 단일 GPU사용을 전제
    """
    
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            scatter(input[i], [devices[i // chunk_size]],
                    [streams[i // chunk_size]]) for i in range(len(input))
        ]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        if devices != [-1]:
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=True)

        return output
    else:
        raise Exception(f'Unknown type {type(input)}.')
    
    
def get_input_device(input):
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
        if isinstance(obj, torch.Tensor):
            assert target_gpus == [-1], "Use only GPU, not CPU"
            return OrigScatter.apply(target_gpus, None, dim, obj)
        
        # if isinstance(obj, DataContainer):      # TODO : DataContainer선언시 
        #     if obj.cpu_only:
        #     return obj.data
        # else:
        #     return Scatter.forward(target_gpus, obj.data)
        
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for targets in target_gpus]
        
        
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None
            
            
def scatter_kwargs(kwargs, target_gpus = [0], dim=0):
        """Scatter with support for kwargs dictionary."""
        inputs = parallel_scatter(inputs, target_gpus, dim) if inputs else []
        kwargs = parallel_scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs


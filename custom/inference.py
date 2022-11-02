import numpy as np
import torch

from transforms.utils import replace_ImageToTensor
from transforms.compose import Compose
from datasets.dataloader import collate
from utils.scatter import parallel_scatter

def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False
        
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    
   
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    

    test_pipeline = Compose(cfg.data.test.pipeline)
    
    datas = []
    for img in imgs:
        # prepare data
        data = dict(img_info=dict(filename=img), img_prefix=None)
        
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)
        
    
    # just get the actual data from DataContainer
    data = collate(datas, samples_per_gpu=len(imgs))
    
    # len: 1
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    
    assert next(model.parameters()).is_cuda, f"modules must be is_cuda, but is not"
    # scatter to specified GPU
    data = parallel_scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
        
    if not is_batch:
        return results[0]
    else:
        return results
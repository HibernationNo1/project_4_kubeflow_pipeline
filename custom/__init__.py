# 여기에 file을 import하고, __init__을 train.py에서 import하면 registry에 포함시킬 수 있다.
import builder
import transforms.compose
import models.swin_transformer.swin_transformer as swin_transformer
import utils.registry as registry
from models.maskrcnn import maskrcnn
# 여기에 file을 import하고, __init__을 train.py에서 import하면 registry에 포하미킬 수 있다.
import builder
import compose
import swin_transformer
import utils.registry as registry
from maskrcnn import maskrcnn
from aimtools.config import *
from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device

import os

def get_model():
# Load model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    half = False
    device = select_device(' ') # device设备，默认为GPU
    data = os.path.join(current_dir, 'else', 'data.yaml')
    imgsz=(640, 640)
    model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    return model, device, half, stride, imgsz, names


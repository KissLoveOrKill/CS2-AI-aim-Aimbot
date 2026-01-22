import os

current_dir = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(current_dir, 'weights', 'headonly.pt')  # 训练出的权重文件
MAX_DET = 1000
IMGSZ = [640, 640]
CONF_THRES = 0.5 #置信度
IOU_THRES = 0.45 # 框重合情况的选择
LINE_THICKNESS = 1 # 框的粗细
HIDE_CONF = True # 是否隐藏标签等
HIDE_LABELS = True # 同上
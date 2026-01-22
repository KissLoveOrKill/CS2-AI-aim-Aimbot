import mss
import numpy as np
import cv2
import torch
import win32gui
import win32con
import pynput
from scipy.constants import pt
from load_model import get_model
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors
from aimtools.config import *

sct = mss.mss()
screen_width = 2560 #屏幕的宽
screen_height = 1600 #屏幕的高
GAME_X = screen_width // 3
GAME_Y = screen_height // 3
GAME_LEFT, GAME_TOP, GAME_WIDTH, GAME_HEIGHT =7 * screen_width // 16, 7 * screen_height // 16, screen_width // 8, screen_height // 8#游戏内截图区域
RESIZE_WINDOW_WIDTH, RESIZE_WINDOW_HEIGHT = screen_width // 8, screen_height // 8 #显示窗口大小
monitor = {
    'left' : GAME_LEFT,
    'top' : GAME_TOP,
    'width' : GAME_WIDTH,
    'height' : GAME_HEIGHT // 3
}
window_name = 'test'
model, device, half, stride, imgsz, names = get_model() # 模型加载

#图像识别
#不进行梯度处理
@torch.no_grad()
def pred_img(im0):
    # Padded resize
    im = letterbox(im0, imgsz, stride=stride, auto=True)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup


    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    # 归一化处理
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False, visualize=False)
    # NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, None, False, max_det=MAX_DET)
    # Process predictions

    # im0 = im0.copy()
    det = pred[0]
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(names))
    xywh_list= []
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            xywh_list.append(xywh)
            c = int(cls)  # integer class
            label = None if HIDE_LABELS else (names[c] if HIDE_CONF else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    # Stream results
    im0 = annotator.result()
    return im0, xywh_list

while True:
    img = sct.grab(monitor) # 抓取屏幕
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 将图片通道类型转为BGR
#函数调用
    img, aims = pred_img(img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) #根据窗口有关比例
    cv2.resizeWindow(window_name, RESIZE_WINDOW_WIDTH, RESIZE_WINDOW_HEIGHT)
    cv2.imshow(window_name, img)

    hwnd = win32gui.FindWindow(None, window_name)
    # 窗口需要正常大小且在后台，不能最小化
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    # 窗口需要最大化且在后台，不能最小化
    # ctypes.windll.user32.ShowWindow(hwnd, 3)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)

    k = cv2.waitKey(1)
    if k % 256 == 27: # 按EAC健退出
        cv2.destroyAllWindows()
        exit('ESC...')




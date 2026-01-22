import mss
import numpy as np
import cv2
import torch
import win32api
import win32gui
import win32con
import pynput
from pynput import mouse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.constants import pt

from aimtools.mouse import move_mouse
from load_model import get_model
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.plots import Annotator, colors
from aimtools.config import *

sct = mss.mss()
screen_width = 2560 #屏幕的宽
screen_height = 1600 #屏幕的高
GAME_LEFT, GAME_TOP, GAME_WIDTH, GAME_HEIGHT =6 * screen_width // 16 , 7 * screen_height // 16, screen_width // 4, screen_height // 4#游戏内截图区域
RESIZE_WINDOW_WIDTH, RESIZE_WINDOW_HEIGHT = screen_width // 4, screen_height // 4 #显示窗口大小
monitor = {
    'left' : GAME_LEFT,
    'top' : GAME_TOP,
    'width' : GAME_WIDTH,
    'height' : GAME_HEIGHT // 3
}
window_name = 'detect'
model, device, half, stride, imgsz, names = get_model() # 模型加载

#加载鼠标控制
mouse_controller = pynput.mouse.Controller()

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

def mouse_aim_controller(xywh_list, mouse,left, top, width, height):
# 鼠标相对于屏幕的位置
    mouse_x, mouse_y = mouse.position
# 寻找最优的位置
    best_xy = None
    for xywh in xywh_list:
        x, y, _, _ = xywh
    # 还原相对于检测区域的位置
        x *= width
        y *= height
    # 转换坐标系
        x += left
        y += top
        dist = ((x - mouse_x) ** 2 + (y - mouse_y) ** 2) ** .5
        if  not best_xy:
            best_xy = ((x, y), dist)
        else:
            _, old_dist = best_xy
            if dist < old_dist :
                best_xy = ((x, y), dist)
# 获取检测区域大小
    x, y = best_xy[0]
    sub_x, sub_y = x - mouse_x, y - mouse_y
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(sub_x), int(sub_y))


LOCK_AIM = False
def on_click(x, y, button, pressed):
    global LOCK_AIM
    if button == button.middle:
        if pressed :
            LOCK_AIM = not LOCK_AIM
            print('自瞄状态', f"[{LOCK_AIM and '开' or '关'}]")

listener = mouse.Listener(on_click=on_click)
listener.start()

while True:
    img = sct.grab(monitor)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 将图片通道类型转为BGR
#函数调用
    img, aims = pred_img(img)
    if aims and LOCK_AIM:
        mouse_aim_controller(aims, mouse_controller, **monitor)

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
    if k % 256 == 27:
        cv2.destroyAllWindows()
        exit('ESC...')




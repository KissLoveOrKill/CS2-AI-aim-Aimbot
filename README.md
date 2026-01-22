# Aimbot Version  - 项目说明文档

## 1. 项目简介 (Introduction)
本项目是一个基于计算机视觉的自动瞄准辅助工具（Aimbot），核心算法采用 **YOLOv5** 目标检测模型。系统通过实时采集屏幕画面，识别游戏中的目标（如 CS 中的角色），并计算坐标差异，驱动鼠标进行自动定位。

本项目基于开源教程开发，并在此基础上进行了自定义改进和功能封装。

## 2. 详细项目结构 (Detailed Project Structure)

```plaintext
aimbotdata/
├── README.md                   # 项目说明文档
├── aimbot_version2/            # 项目主目录
│   └── yolov5-7.0/             # YOLOv5 源码与核心逻辑
│       ├── aimtools/           # [核心] 自定义功能模块
│       │   ├── config.py       # 全局配置文件 (参数设置)
│       │   ├── grab_screen.py  # 屏幕截图模块
│       │   ├── load_model.py   # 模型加载模块
│       │   ├── main.py         # [启动入口] 主程序 (原 mian.py)
│       │   ├── mouse_control.py# 鼠标控制逻辑
│       │   ├── else/           # 包含 data.yaml 等配置
│       │   └── weights/        # 存放推理用的权重文件 (headonly.pt)
│       ├── detect.py           # YOLOv5 原生检测脚本
│       ├── train.py            # YOLOv5 训练脚本
│       ├── requirements.txt    # 项目依赖列表
│       └── ...
├── 数据集/                     # 训练数据集
│   ├── README.roboflow.txt     # [数据源] https://universe.roboflow.com/aimbot-dubsm/aimbot-2-vmxku
│   ├── data.yaml               # 数据集配置文件
│   ├── train/                  # 训练集图片与标签
│   ├── valid/                  # 验证集图片与标签
│   └── ...
└── 训练结果/                   # 模型训练产物
    ├── results.csv             # 训练指标数据
    ├── weights/                # 训练好的权重
    │   ├── best.pt             # 最佳权重
    │   └── last.pt             # 最终权重
    └── ...
```

## 3. 核心技术栈 (Tech Stack)
*   **目标检测**: YOLOv5 (PyTorch) - 深度学习模型
*   **屏幕采集**: MSS (Multiple Screen Shots) - 高帧率屏幕录制
*   **输入控制**: win32api / pynput - 模拟鼠标移动与监听输入
*   **图像处理**: OpenCV / NumPy - 图像数据处理与可视化

## 4. 详细使用方法 (User Guide)

### 4.1 环境准备
1.  确保已安装 **Python 3.8+**。
2.  (可选) 建议安装 CUDA 和 cuDNN 以利用 NVIDIA GPU 进行加速，获得更高的帧率。

### 4.2 安装依赖
在终端中进入 `aimbot_version2/yolov5-7.0` 目录，安装所需的 Python 库：

```bash
cd d:\aimbotdata\aimbot_version2\yolov5-7.0
pip install -r requirements.txt
```

### 4.3 配置文件设置
在运行前，请检查 `aimtools/config.py` 和 `aimtools/main.py` 中的设置，以适配你的显示器和游戏习惯：

*   **`aimtools/main.py`**:
    *   `screen_width`, `screen_height`: 修改为你的屏幕分辨率（当前默认为 2560x1600）。
    *   `GAME_LEFT`, `GAME_TOP`, ...: 截图区域设置，默认截取屏幕中心区域。
*   **`aimtools/config.py`**:
    *   `CONF_THRES`: 置信度阈值（默认 0.5），调低可增加识别率，但可能增加误报。
    *   `IOU_THRES`: IOU 阈值。

### 4.4 运行项目
使用以下命令启动自动瞄准主程序：

```bash
cd d:\aimbotdata\aimbot_version2\yolov5-7.0
python aimtools/main.py
```

### 4.5 操作说明与快捷键
程序启动后，会显示一个名为 `detect` 的预览窗口，实时显示检测结果。

*   **开启/关闭自瞄**: 点击 **鼠标中键** (Middle Mouse Button)。
    *   控制台会输出状态：`自瞄状态 [开]` 或 `自瞄状态 [关]`。
*   **退出程序**: 按下 **ESC** 键。

### 4.6 注意事项
*   请确保游戏以 **窗口化** 或 **无边框窗口** 模式运行，全屏模式下可能无法正常截图或绘制预览。
*   部分游戏反作弊系统可能检测到模拟鼠标输入或 Hook 行为，请自行承担使用风险。建议仅在单机模式或允许的环境下测试。

 **参考来源**: 本项目主要流程参考了 Bilibili UP主 木-酥 的教学视频。


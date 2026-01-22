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

## 4. 核心组件详解 (Core Components)

为了帮助理解代码逻辑，以下是 `aimtools` 下各核心模块的详细说明：

### 4.1 程序入口 (`main.py`)
这是整个系统的**控制中心**。它初始化所有其他模块，并维持一个无限循环来处理实时任务。
*   **职责**：协调截屏、推理、控制三者的运行时序。
*   **交互**：通过 `win32api` 监听按键（默认鼠标中键），实现“按需启动”的自瞄逻辑，防止误操作。

### 4.2 模型加载器 (`load_model.py`)
负责将训练好的权重文件（`.pt`）加载到内存中。
*   **智能加速**：会自动检测当前环境是否支持 **CUDA**。如果检测到 NVIDIA 显卡，会自动启用 GPU 加速推理，这对于实现高 FPS（帧率）至关重要。

### 4.3 极速截屏 (`grab_screen.py`)
利用 **MSS (Multiple Screen Shots)** 库实现的屏幕采集模块。
*   **性能优势**：相比传统的截图方式，MSS 直接从操作系统底层获取显存数据，延迟极低。
*   **区域锁定**：只截取屏幕中心（准星周围）的特定区域进行识别，大幅减少了计算量。

### 4.4 鼠标控制器 (`mouse_control.py`)
系统的“执行手”。
*   **坐标计算**：接收 YOLO 识别出的目标框，计算目标中心点与当前准星的相对距离。
*   **动作执行**：调用 Windows 底层 API (`win32api.mouse_event`) 移动鼠标。

### 4.5 全局配置 (`config.py`)
存放项目的所有关键参数，无需修改代码即可调整配置。
*   **参数示例**：包括模型路径、置信度阈值 (`conf_thres`)、IOU 阈值等。

## 5. 详细使用方法 (User Guide)

### 5.1 环境准备
1.  确保已安装 **Python 3.8+**。
2.  (可选) 建议安装 CUDA 和 cuDNN 以利用 NVIDIA GPU 进行加速，获得更高的帧率。

### 5.2 安装依赖
在终端中进入 `aimbot_version2/yolov5-7.0` 目录，安装所需的 Python 库：

```bash
cd d:\aimbotdata\aimbot_version2\yolov5-7.0
pip install -r requirements.txt
```

### 5.3 详细配置指南 (Configuration Guide)
为了获得最佳的识别效果和帧率，**必须**根据您的硬件环境修改以下配置。

#### A. 屏幕与区域设置 (`aimtools/main.py`)
主要用于适配您的显示器分辨率和截图范围。

| 变量名 | 默认值 | 作用与修改建议 |
| :--- | :--- | :--- |
| **`screen_width`** | `2560` | **显示器物理宽度**。请务必修改为您当前屏幕的分辨率（如 1920）。 |
| **`screen_height`** | `1600` | **显示器物理高度**。请务必修改为您当前屏幕的分辨率（如 1080）。 |
| `GAME_WIDTH` | `screen_width // 4` | **水平检测范围**。当前设为屏幕宽度的 1/4。数值越小，检测范围越小，但 FPS 越高。 |
| `GAME_HEIGHT` | `screen_height // 4` | **垂直检测基准**。用于定义垂直方向的截图基准大小。 |
| `monitor['height']` | `GAME_HEIGHT // 3` | **实际截图高度**。代码中默认仅截取垂直区域的 1/3（准星附近的一条细长区域），以极致压缩计算量。 |

#### B. 算法敏感度设置 (`aimtools/config.py`)
主要用于平衡“识别准确率”和“误报率”。

| 变量名 | 默认值 | 作用与修改建议 |
| :--- | :--- | :--- |
| **`CONF_THRES`** | `0.5` | **置信度阈值 (0-1)**。表示机器认为“这是一个目标”的最低信心。 <br>• **调低 (如 0.3)**: 能识别更远、更模糊的敌人，但容易把箱子/队友误认为敌人。<br>• **调高 (如 0.7)**: 只有非常清晰时才开枪，甚至可能漏掉敌人。 |
| `IOU_THRES` | `0.45` | **重叠过滤阈值**。用于去除针对同一目标的重复检测框，通常无需修改。 |
| `WEIGHTS` | `.../headonly.pt` | **模型路径**。如果您训练了自己的模型（如 `best.pt`），请修改此处的文件名为您的模型名称。 |
| `HIDE_LABELS` | `True` | **隐藏标签文字**。设为 `True` 时预览窗口不显示 "head" 字样，只显示框，视野更清晰。 |
| `LINE_THICKNESS`| `1` | **框线粗细**。预览窗口中红色框的粗细，数值越大框越粗。 |

### 5.4 运行项目
使用以下命令启动自动瞄准主程序：

```bash
cd d:\aimbotdata\aimbot_version2\yolov5-7.0
python aimtools/main.py
```

### 5.5 操作说明与快捷键
程序启动后，会显示一个名为 `detect` 的预览窗口，实时显示检测结果。

*   **开启/关闭自瞄**: 点击 **鼠标中键** (Middle Mouse Button)。
    *   控制台会输出状态：`自瞄状态 [开]` 或 `自瞄状态 [关]`。
*   **退出程序**: 按下 **ESC** 键。

### 5.6 注意事项
*   请确保游戏以 **窗口化** 或 **无边框窗口** 模式运行，全屏模式下可能无法正常截图或绘制预览。
*   部分游戏反作弊系统可能检测到模拟鼠标输入或 Hook 行为，请自行承担使用风险。建议仅在单机模式或允许的环境下测试。

 **参考来源**: 本项目主要流程参考了 Bilibili UP主 木-酥 的教学视频
 （https://space.bilibili.com/108615693?spm_id_from=333.337.0.0）


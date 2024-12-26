# LivePortrait Warping Module 权重转换与推理加速

本项目旨在将 LivePortrait 中的 warping_module 模型进行转换并优化，最终实现推理加速。通过使用 AutoDL 平台进行模型的转换（从 PyTorch 到 ONNX，再到 TensorRT），以及基于 AutoDL 环境进行推理加速，从而提升性能。

## 项目简介

LivePortrait 是一款用于实时人脸图像处理的工具，其中的 warping_module 负责图像的几何变换和空间扭曲。为了在 AutoDL 环境下加速推理，我们首先将该模块转换为 ONNX 格式，接着将其转为 TensorRT 格式以便进行硬件加速。

AutoDL 提供了丰富的深度学习实验平台，便于管理实验过程、模型转换及推理过程，并且能够通过 NVIDIA GPU 进行性能优化。通过在 AutoDL 环境中执行这些任务，能够实现更高效的深度学习推理。

## 环境要求

- **Python 3.10或更高版本**
- **PyTorch**（用于训练和转换模型）
- **ONNX**（用于转换模型到 ONNX 格式）
- **TensorRT=10.7.0**（用于模型加速）
- **CUDA=12.1**（推荐使用支持的 NVIDIA GPU）

可以通过以下命令安装依赖：

```
git clone https://github.com/jinanfeng74/Inference_Acceleration_LivePortrait.git
# create env using conda
conda create -n LivePortrait python==3.10
conda activate LivePortrait
# install dependencies with pip
pip install -r requirements.txt
sudo apt install ffmpeg
```

## 步骤

## 1. 模型转换为 ONNX 格式

首先，我们需要将 **LivePortrait** 中的 `warping_module` 模型转换为 ONNX 格式。将下载好的的 PyTorch 模型，可以通过script脚本进行转换：

python pth2onnx.py
执行成功之后就可以在script/warping_module.onxx/目录下看到转换后的onnx文件
## 2. 将 ONNX 模型转换为 TensorRT

在将模型导出为 ONNX 格式后，我们可以使用 TensorRT 进行进一步的加速。将 ONNX 模型转换为 TensorRT 格式：
# Real-time-portrait-acceleration

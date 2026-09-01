# MNIST Simple

一个基于 PyTorch 的入门级 MNIST 手写数字识别项目。

项目包含两个版本：

- `mnist_simple.py`：带日志输出，训练过程会写入 `output.log`
- `mnist_simple_v2.py`：更简洁的训练脚本，直接在终端输出结果

## 项目目标

使用一个简单的全连接神经网络，对 MNIST 数据集中的 0~9 手写数字进行分类，并完成：

- 数据下载与预处理
- 模型定义
- 训练
- 测试与准确率评估

## 环境要求

- Python 3.9+
- PyTorch
- torchvision

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

训练并测试第一个版本：

```bash
python mnist_simple.py
```

训练并测试第二个版本：

```bash
python mnist_simple_v2.py
```

## 目录说明

- `mnist_simple.py`：基础版本，带 `output.log` 记录
- `mnist_simple_v2.py`：精简版本
- `data/`：MNIST 数据集下载目录，首次运行时自动生成
- `output.log`：`mnist_simple.py` 运行时生成的日志文件

## 模型结构

两个脚本都使用了同一个基础网络：

- 输入层：`28 x 28 = 784`
- 隐藏层：`784 -> 128`
- 输出层：`128 -> 10`
- 激活函数：ReLU

## 说明

- 数据集会通过 `torchvision` 自动下载
- 如果你已经下载过数据，脚本会直接读取本地 `data/` 目录
- 训练轮数在脚本中可以直接修改

## 适合作为展示的内容

这个项目适合展示你对以下内容的理解：

- PyTorch 基础训练流程
- MNIST 图像分类任务
- DataLoader、loss、optimizer 的使用
- 训练/测试模式切换


# ============================================================
# 极简版 MNIST 手写数字识别 v2
# 功能：下载数据 → 定义模型 → 训练 → 测试准确率
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── 第一步：数据预处理 & 加载 ──────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),                        # 图片 → 张量（0~1）
    transforms.Normalize((0.1307,), (0.3081,))    # 标准化（均值、标准差）
])

train_loader = DataLoader(
    datasets.MNIST('./data', train=True,  download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=64, shuffle=False
)

print(f"数据加载完成：训练集 {len(train_loader.dataset)} 张，测试集 {len(test_loader.dataset)} 张")

# ── 第二步：定义神经网络 ──────────────────────────────────────
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()                        # 初始化父类（简写）
        self.fc1 = nn.Linear(784, 128)            # 输入层 → 隐藏层
        self.fc2 = nn.Linear(128, 10)             # 隐藏层 → 输出层（10个数字）
        self.relu = nn.ReLU()                     # 激活函数（增加非线性）

    def forward(self, x):
        x = x.view(-1, 784)                       # 28×28 图片拉平成 784 个数
        x = self.relu(self.fc1(x))                # 第一层 + 激活
        return self.fc2(x)                        # 第二层，输出10个分数

# ── 第三步：(准备训练) 初始化模型、损失函数、优化器 ─────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()          # 损失函数（衡量预测有多错）
optimizer = optim.Adam(model.parameters(), lr=0.001)    # 优化器（负责更新参数）
# lr=0.001 是学习率，代表每次更新参数的"步子"有多大

print(f"使用设备：{device}")

# ── 第四步：训练模型 ──────────────────────────────────────────
EPOCHS = 10  #把整个训练集完整过 3 遍    #训练轮数（把60000张图完整过一遍 = 1轮）

for epoch in range(EPOCHS):
    model.train()                #把模型设置为训练模式
    total_loss = 0      #用来记录当前 epoch 所有批次的损失总和，后面用来算平均损失

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  #把数据移动到 device上，和模型在同一个设备，才能计算
        # images: 当前批次的图片张量，形状 (64, 1, 28, 28)
        # labels: 当前批次的标签（正确答案），形状 (64,)
        optimizer.zero_grad()              # 清零梯度  (每批次前先清零梯度（固定写法）)
        output = model(images)             # 前向传播 (把图片喂给模型，得到预测分数)
        loss = criterion(output, labels)   # 计算损失（预测 vs 正确答案）
        loss.backward()                    # 反向传播（计算梯度）
        optimizer.step()                   # 更新参数（根据梯度调整权重）

        total_loss += loss.item()          # 累计损失
        # 每隔200个批次打印一次进度
        if batch_idx % 200 == 0:
            print(f"  轮次 {epoch+1}/{EPOCHS} | 批次 {batch_idx}/{len(train_loader)} | 损失: {loss.item():.4f}")

    print(f"第 {epoch+1} 轮完成，平均损失: {total_loss / len(train_loader):.4f}")

# ── 第五步：测试准确率 ────────────────────────────────────────
model.eval()       # 切换到测试模式（关闭Dropout等）
correct = 0        # 预测正确的数量

with torch.no_grad():                         # 测试时不需要计算梯度，节省内存
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images), 1)       # _ 丢弃最大值，只保留最大值所在位置（即预测的数字） # 取分数最高的那个数字作为预测结果
        correct += (predicted == labels).sum().item()      # 累计正确数

total = len(test_loader.dataset)
print(f"\n测试结果：{correct}/{total} 预测正确")
print(f"准确率：{100 * correct / total:.2f}%")
print("训练完成！")

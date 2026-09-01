# ============================================================
# 极简版 MNIST 手写数字识别
# 作者：初学者练习版
# 功能：下载数据 → 定义模型 → 训练 → 测试准确率
# ============================================================

# ── 第一步：导入需要的库 ──────────────────────────────────────
import torch                                      # PyTorch 主库
import torch.nn as nn                             # 神经网络模块（定义层）
import torch.optim as optim                       # 优化器（更新参数）
from torchvision import datasets, transforms      # 图像数据集 + 图片处理
from torch.utils.data import DataLoader           # 数据批量加载器
import sys                                        # 用于同时输出到日志文件

# ── 日志：同时输出到屏幕和 output.log ──────────────────────────
log_file = open('output.log', 'w', encoding='utf-8')

def log(msg):
    print(msg)             # 1. 信息打印到屏幕
    log_file.write(str(msg) + '\n')      # 2. 信息写入日志文件 | 文件的 write () 方法只接受字符串！
    log_file.flush()            # 3. 强制立刻写入硬盘，不缓存


# ── 第二步：下载并加载数据 ────────────────────────────────────
# 定义图片处理流程：先转张量，再标准化
transform = transforms.Compose([
    transforms.ToTensor(),                        # 把图片变成张量（0~1之间的小数）
    transforms.Normalize((0.1307,), (0.3081,))    # 标准化（均值、标准差）
])

# 下载训练集（60000张）
train_dataset = datasets.MNIST(
    root='./data',       # 数据存放在当前目录的 data 文件夹
    train=True,          # True = 训练集
    download=True,       # 自动下载（本地有就直接用）
    transform=transform  # 应用上面定义的处理流程
)

# 下载测试集（10000张）
test_dataset = datasets.MNIST(
    root='./data',
    train=False,         # False = 测试集
    download=True,
    transform=transform
)

# 用 DataLoader 把数据包装成"一批一批"的形式
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# batch_size=64：每次喂给模型64张图
# shuffle=True：每轮训练前打乱顺序

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

log(f"数据加载完成：训练集 {len(train_dataset)} 张，测试集 {len(test_dataset)} 张")


# ── 第三步：定义神经网络 ──────────────────────────────────────
class SimpleNet(nn.Module):                       # 自定义网络，继承 nn.Module
    def __init__(self):
        super(SimpleNet, self).__init__()         # 固定写法，初始化父类
        # 全连接层 1：把 784 个像素 → 压缩成 128 个特征
        self.fc1 = nn.Linear(784, 128)
        # 全连接层 2：把 128 个特征 → 输出 10 个分数（对应数字0~9）
        self.fc2 = nn.Linear(128, 10)
        # 激活函数：给网络增加"非线性"，让它学到更复杂的特征
        self.relu = nn.ReLU()

    def forward(self, x):                         # 定义数据流动的方向
        x = x.view(-1, 784)                       # 把图片从(1,28,28)拉平成784个数
        x = self.relu(self.fc1(x))                # 经过第一层 + 激活函数
        x = self.fc2(x)                           # 经过第二层，输出10个分数
        return x


# ── 第四步：准备训练 ──────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)                    # 创建模型，放到 GPU 或 CPU
criterion = nn.CrossEntropyLoss()                 # 损失函数（衡量预测有多错）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器（负责更新参数）
# lr=0.001 是学习率，代表每次更新参数的"步子"有多大

log(f"使用设备：{device}")


# ── 第五步：训练模型 ──────────────────────────────────────────
EPOCHS = 3  # 训练轮数（把60000张图完整过一遍 = 1轮）

log("=" * 50)
log("开始训练...")
log("=" * 50)

for epoch in range(EPOCHS):                       # 训练 3 轮
    model.train()                                 # 切换到训练模式
    total_loss = 0                                # 记录这一轮的总损失

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: 当前批次的图片张量，形状 (64, 1, 28, 28)
        # labels: 当前批次的标签（正确答案），形状 (64,)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                     # 每批次前先清零梯度（固定写法）
        output = model(images)                    # 把图片喂给模型，得到预测分数
        loss = criterion(output, labels)          # 计算损失（预测 vs 正确答案）
        loss.backward()                           # 反向传播（计算梯度）
        optimizer.step()                          # 更新参数（根据梯度调整权重）

        total_loss += loss.item()                 # 累计损失

        # 每隔200个批次打印一次进度
        if batch_idx % 200 == 0:
            log(f"  轮次 {epoch+1}/{EPOCHS} | 批次 {batch_idx}/{len(train_loader)} | 损失: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    log(f"第 {epoch+1} 轮完成，平均损失: {avg_loss:.4f}")
    log("-" * 50)


# ── 第六步：测试准确率 ────────────────────────────────────────
log("\n开始测试...")
model.eval()                                      # 切换到测试模式（关闭Dropout等）

correct = 0   # 预测正确的数量
total = 0     # 总图片数量

with torch.no_grad():                             # 测试时不需要计算梯度，节省内存
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)                    # 得到预测分数
        _, predicted = torch.max(output, 1)       # 取分数最高的那个数字作为预测结果
        total += labels.size(0)                   # 累计总数
        correct += (predicted == labels).sum().item()  # 累计正确数

accuracy = 100 * correct / total
log(f"\n测试结果：{correct}/{total} 预测正确")
log(f"准确率：{accuracy:.2f}%")
log("\n训练完成！")

log_file.close()

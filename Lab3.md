# 作业3 卷积神经网络进阶

## 一、深度卷积神经网络AlexNet

### 1.1 准备工作
* 本题的AlexNet要求使用`pytorch`深度学习框架完成；程序的设计应包括：**模型设计、数据预处理、模型训练**
* 实验环境：`Anaconda`、`python 3.11`、`pytorch`、`Jupyter Notebook`、CPU

### 1.2 模型设计
本题已经给出深度学习模型的卷积层，只需要用`pytorch`框架将模型表述出来即可

```
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第二个卷积层
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第三个卷积层
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第四个卷积层
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第五个卷积层
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

以第一个卷积层为例：

```
            # 第一个卷积层
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
```

* `3`表示输入通道为3
* `96`表示输出通道为96
* `kernel_size=11`表示卷积核的大小为11*11
* `stride=4`表示卷积操作的步幅为4
* `padding=2`表示边缘补上2像素的0
* `nn.MaxPool2d(kernel_size=3, stride=2)`表示3*3最大汇聚层，步幅为2

### 1.3 数据预处理
```
from torchvision import datasets, transforms

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 加载 Fashion-MNIST 数据集
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

# 数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False
)
```

### 1.4 训练

#### 模型初始化
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### 训练
```
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    print(f"Epoch {epoch} Training Loss: {total_loss / len(train_loader):.6f}")
```

#### 执行训练的结果

<img width="372" alt="{DDE4C1DA-CB2D-4E2B-AA4E-79CA78FD5E47}" src="https://github.com/user-attachments/assets/da9ff5c6-5c74-4c63-8269-3ff4a2b41884">

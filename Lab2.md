# 卷积神经网络基础

## 1. 卷积神经网络范例

### 1.1 数据预处理
将数据归一化到[-1,1]
```
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

### 1.2 定义卷积神经网络模型
```
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入1个通道，输出32个通道
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64个通道
        self.fc2 = nn.Linear(128, 10)  # 输出10个类别

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 1.3 初始化模型，损失函数，优化器
```
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 1.4 训练模型
```
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()  
            outputs = model(images)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  # 更新参数

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

### 1.5 评估模型
```
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
```

### 1.6 评估结果
<img width="310" alt="{92627B05-E320-4687-B19B-723657CD9049}" 
  src="https://github.com/user-attachments/assets/12e0b4e2-c131-4a72-879d-66c7d75cbf5e">

## 2. LeNet卷积神经网络

本神经网络模型在基础的神经网络模型的基础上，需要改进模型的结构

```
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2) 
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)       
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)         
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                  
        self.fc2 = nn.Linear(120, 84)                         
        self.fc3 = nn.Linear(84, 10)                            

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))               
        x = self.pool(torch.relu(self.conv2(x)))             
        x = x.view(-1, 16 * 5 * 5)                            
        x = torch.relu(self.fc1(x))                           
        x = torch.relu(self.fc2(x))                           
        x = self.fc3(x)                                        
        return x
```

测试结果：

<img width="366" alt="{C5C045A6-781D-4B3E-9AB5-46772842A0E9}" src="https://github.com/user-attachments/assets/e730352a-efb6-482e-99e0-42f71eac57e6">

## 3. 实现卷积与池化操作

### 3.1 卷积

#### 3.1.1 初始化
```
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.random.randn(out_channels)
```

#### 3.1.2 向前传播
```
  def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size + 2 * self.padding) 
        out_width = (in_width - self.kernel_size + 2 * self.padding) 

        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # 零填充
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        output[b, oc, h, w] = np.sum(x_padded[b, :, h_start:h_end, w_start:w_end] * self.weight[oc]) + self.bias[oc]
        
        return output
```

### 3.2 池化
```
class MaxPool:
    def __init__(self, kernel_size, stride=1):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        output = np.zeros((batch_size, in_channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        output[b, c, h, w] = np.max(x[b, c, h_start:h_end, w_start:w_end])

        return output
```

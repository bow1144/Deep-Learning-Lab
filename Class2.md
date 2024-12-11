# 实验二 实现反向传播算法

## 一、实验目的
>  在实验二的基础上，学会实现一个基于Numpy的反向传播算法，
> 用以替换实验二中基于数值微分梯度下降算法，从而深入理解和掌握反向传播算法的原理。

## 二、实验开发环境
> Anaconda, Python 3.6.8, Pytorch

## 三、实验内容

### 3.1 ReLU激活层
```
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
    #获取x数组中小于0的元素的索引
        self.mask = (x <= 0)
        out = x.copy()    #out变量表示要正向传播给下一层的数据，即上图中的y
        ###请补充代码将x数组中小于0的元素赋值为0
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        ###请补充代码完成Relu层的反向传播
        return dout
```
* `self.mask = (x <= 0)`表示创建一个布尔数组，如果这个元素的值小于等于0则为True
* `out[self.mask] = 0`将小于等于0的数置为0
* `return dout`直接将置为0的数组返回

### 3.2 全连接层的实现
```
class Linear:
    def __init__(self, W, b):
        self.W = W # 权重参数
        self.b = b # 偏置参数
        self.x = None # 用于保存输入数据
        # 定义成员变量用于保存权重和偏置参数的梯度
        self.dW = None
        self.db = None

        #全连接层的前向传播
    def forward(self, x):
        #保存输入数据到成员变量用于backward中的计算
        self.x = x
        ###请补充代码求全连接层的前向传播的输出保存到变量out中
        out = x.dot(self.W) + self.b
        return out
    
        #全连接层的反向传播
    def backward(self, dout):
        ###请同学补充代码完成求取dx,dw,db，dw,db保存到成员变量self.dW,self.db中
        dx = dout.dot(self.W.T)
        self.dW = self.x.T.dot(dout)
        self.db = dout.sum(axis=0)

        return dx
```
* `out = x.dot(self.W) + self.b`计算线性层的输出
* `dx = dout.dot(self.W.T)`根据反向传播的链式法则，由上一层传下来的梯度乘以W的转置

### 3.3 Softmax-with-Loss层的实现
```
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    # SoftmaxWithLoss层的前向传播函数
    def forward(self, x, t):
        self.t = t
        x -= np.max(x, axis=1, keepdims=True)  # 防止溢出
        self.y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)  # 计算softmax输出
        
        epsilon = 1e-7
        self.y = np.clip(self.y, epsilon, 1 - epsilon)  # 限制
    
        if self.t.ndim == 1:
            self.t = np.eye(self.y.shape[1])[self.t]  # 将t转换为one-hot编码
            
        self.loss = -np.sum(np.log(self.y) * self.t) / x.shape[0]  # 计算交叉熵损失
        return self.loss


    # SoftmaxWithLoss层的反向传播函数
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # 计算反向传播的梯度
        return dx
```
* 计算`self.y`，简单来说就是`self.y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)`的
  SoftMax算法，但是为了防止指数对数计算的错误加入小变量
* 将数据转化为*one-hot*编码：`self.t = np.eye(self.y.shape[1])[self.t]`
* ` self.loss = -np.sum(np.log(self.y) * self.t) / x.shape[0] `计算交叉熵损失
* `dx = (self.y - self.t) / batch_size`是反向传播中计算的梯度

### 3.4 模型构建
```
class TwoLayerNet:
    #模型初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        #获取第一层权重和偏置
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        #获取第二层权重和偏置
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        #将神经网络的层保存为有序字典OrderedDict
        self.layers = OrderedDict()
        #添加第一个全连接层到有序字典中
        self.layers['Linear1'] = Linear(self.params['W1'], self.params['b1'])
        # 添加ReLU层
        self.layers['ReLU1'] = Relu()
        # 添加第二个全连接层
        self.layers['Linear2'] = Linear(self.params['W2'], self.params['b2'])

        #将SoftmaxWithLoss类实例化为self.lastLayer
        self.lastLayer = SoftmaxWithLoss()

    #通过前向传播获取预测值
    def predict(self, x):
        #遍历有序字典
        for layer in self.layers.values():
            x = layer.forward(x)  # 进行前向传播

        return x

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        #获取预测值
        y = self.predict(x)
        #返回损失
        return self.lastLayer.forward(y, t)

    #求精度
    def accuracy(self, x, t):
        y = self.predict(x)  # 使用预测方法获取 y
        # 如果 t 是一维数组（标签索引），将其转换为 one-hot 编码
        if t.ndim == 1:
            t = np.eye(y.shape[1])[t]  # 使用 y 的输出维度来生成 one-hot 编码
        y = np.argmax(y, axis=1)  # 获取预测值的最大概率索引
        t = np.argmax(t, axis=1)  # 获取真实标签的最大索引
        accuracy = np.sum(y == t) / float(x.shape[0])  # 计算准确率
        return accuracy

    #求梯度
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1  # SoftmaxWithLoss 的反向传播输入是 1（通常是损失函数的梯度）

        # 反向传播 SoftmaxWithLoss
        dout = self.lastLayer.backward(dout)

        # 从后往前遍历有序字典
        layers = list(self.layers.values())
        layers.reverse()  # 倒序处理，最后一个层首先处理
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        # 获取第一层网络参数的梯度
        grads['W1'], grads['b1'] = self.layers['Linear1'].dW, self.layers['Linear1'].db
        # 获取第二层网络参数的梯度
        grads['W2'], grads['b2'] = self.layers['Linear2'].dW, self.layers['Linear2'].db

        return grads
```
* 在第一个全连接层和第二个全连接层之间加入`ReLU`激活函数
* 遍历有序字典，并计算向前传播`x = layer.forward(x) `
* 修改：将t改成`one-hot`矩阵，目的是方便维度统一
* 倒序处理梯度`dout = layer.backward(dout)`，首先计算`SoftmaxWithLoss`的梯度，再计算前面
  全连接层和激活函数层的梯度

### 3.5 训练
> 代码和数据仿照第一个实验

```
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [0, 1]
])

# 下载 MNIST 数据集并加载到训练集
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 获取训练集的图像和标签
x_full_train = full_train_dataset.data.numpy()  # 原始图像数据，shape = (60000, 28, 28)
t_full_train = full_train_dataset.targets.numpy()  # 原始标签数据，shape = (60000,)

# 将图像数据展平为 784 维的向量，shape = (60000, 784)
x_full_train = x_full_train.reshape(-1, 784)

# 按照 82% 和 18% 的比例拆分训练集数据
x_train, x_test, t_train, t_test = train_test_split(x_full_train, t_full_train, test_size=0.18, random_state=42)
```

```
import numpy as np
from sklearn.metrics import accuracy_score

# 定义训练循环的迭代次数
iters_num = 10000  
# 获取训练数据的规模
train_size = x_train.shape[0]
print(f"train_size:{train_size}")
# 定义训练批次大小
batch_size = 100
# 定义学习率
learning_rate = 0.001

# 创建记录模型训练损失值的列表
train_loss_list = []
# 创建记录模型在训练数据集上预测精度的列表
train_acc_list = []
# 创建记录模型在测试数据集上预测精度的列表
test_acc_list = []

# 计算一个epoch所需的训练迭代次数（一个epoch定义为所有训练数据都遍历过一次所需的迭代次数）
iter_per_epoch = max(train_size / batch_size, 1)

# 实例化TwoLayerNet类，生成一个network对象
input_size = 784  # 输入神经元个数（28x28像素的图片）
hidden_size = 50  # 隐藏层神经元个数
output_size = 10  # 输出神经元个数（10类）
network = TwoLayerNet(input_size, hidden_size, output_size)

print(x_train[0].size)

# 创建训练循环
for i in range(iters_num):
    # 在每次训练迭代内部选择一个批次的数据
    batch_mask = np.random.choice(train_size, batch_size)  # 随机选择批次
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # print(f"x_batch.size = {x_batch.size}")
    
    # 计算梯度
    grads = network.gradient(x_batch, t_batch)
    
    # 更新模型参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]

    # 计算损失值
    loss = network.loss(x_batch, t_batch)
    
    # 向train_loss_list添加本轮迭代的损失值
    train_loss_list.append(loss)

    # 判断是否完成了一个epoch（即训练完一个完整的批次）
    if i % iter_per_epoch == 0:
        # 计算训练集上的准确率
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)

        # 计算测试集上的准确率
        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)

        # 输出一个epoch完成后，模型在训练集和测试集上的精度和损失值
        print(f"iteration:{i}, train acc: {train_acc}, test acc: {test_acc}, loss: {loss}")
```

![{9FC3E238-526C-428E-8BA3-14248EDACFE7}](https://github.com/user-attachments/assets/572485d4-e283-45f2-b47d-149e535adf37)

* 速度极快，几乎可以在一分钟之内跑完一万层，并且准确率较高
* 需要修改学习率：如果学习率大于0.05，会因为跨度太大而一直无法成功训练（准确率在12%一下），在将
  学习率调整到0.001后，准确率最高可以达到97%

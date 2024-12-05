# 实验一 基于Numpy实现神经网络

## 一、实验目的 
    学会基于Numpy实现一个简单的全连接前馈神经网络，并用该网络识别手写数字体。
## 二、实验开发环境和工具
可以在Ubuntu18.04操作系统上搭建开发环境，所使用的开发工具为Anaconda，使用Python语言进行开发。

## 三、实验内容

### 3.1 获取数据集
#### 3.1.1 下载数据集
* 使用`torchvision`中的`dataset`下载数据集
```
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
```

#### 3.1.2 获取训练集的图像和标签
```
x_full_train = full_train_dataset.data.numpy()  # 原始图像数据，shape = (60000, 28, 28)
t_full_train = full_train_dataset.targets.numpy()  # 原始标签数据，shape = (60000,)
```

#### 3.1.3 将图像展开为向量
```
x_full_train = x_full_train.reshape(-1, 784)
```

#### 3.1.4 将图像拆分为训练集和测试集
* 按照82%和18%的比例拆分
```
x_train, x_test, t_train, t_test = train_test_split(x_full_train, t_full_train, test_size=0.18, random_state=42)
```

### 3.2 定义激活函数
#### 3.2.1 `sigmoid`函数
* `sigmoid`函数的公式是 $\frac{1}{1+e^{-x}}$ 
```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

#### 3.2.2 `softmax`函数
* `softmax`函数的公式是 $\frac{e^{a_k}}{\sum_i{e^{a_i}}}$
```
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=-1, keepdims=True)
```
* 其中`keepdims`参数的作用是在计算后保留原数组的维度

### 3.3 定义损失函数
```
def cross_entropy_error(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if y_true.ndim == 1:
        y_true = np.eye(y_pred.shape[1])[y_true]
    return - (y_true * np.log(y_pred)).sum(axis=1).mean()
```
* 输入：`y_true`和`y_pred`两个向量
* 交叉信息熵的作用是衡量实际标签与概率分布之间的差异
* 交叉信息熵公式 $H=-\sum_i{y_t(i)\log(y_p(i))}$
* `epsilon = 1e-7`的作用是防止对数计算错误
* `if y_true.ndim == 1:`的作用是转换one-hot编码

### 3.4 获取权重参数梯度
```
def numerical_gradient(f, x):
    epsilon = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = tmp_val + epsilon #+
        f_plus = f(x)
        x[idx] = tmp_val - epsilon #-
        f_minus = f(x)

        grad[idx] = (f_plus - f_minus) / (2 * epsilon) # 按微分定义计算
        x[idx] = tmp_val
        it.iternext()
    return grad
```
* 使用中心差分法计算微分 $\frac{\partial f}{\partial x_i} \approx \frac{f(x_i+\epsilon)-f(x_i-\epsilon)}{2\epsilon}$
* ` it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])`是迭代器，用于遍历x，将数组设置为可写
* `grad = np.zeros_like(x)`创建一个形状类似x的数组，全部设置为0

### 3.5 构建网络
#### 3.5.1 初始化网络
```
def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # 初始化权重
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
```
* 对于权重，使用均值为0的正态分布
* 对于偏置，初始化为0

#### 3.5.2 向前传播
```
def forward(self, x):
    # 获取网络参数
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    # 第一层的运算
    z1 = np.dot(x, W1) + b1 
    h1 = sigmoid(z1) 
    z2 = np.dot(h1, W2) + b2 
    return softmax(z2)
````
* `z1 = np.dot(x, W1) + b1`计算x和w的乘积
* `h1 = sigmoid(z1)`用`sigmoid`函数计算隐藏层

#### 3.5.3 损失函数
```
def loss(self, x, t):
    y = self.forward(x)
    return cross_entropy_error(t, y)
```
* `y = self.forward(x)`是计算的输出向量（10维）
* 计算交叉信息熵

#### 3.5.4 精度计算
```
def accuracy(self, x, t):
    y = self.forward(x) 
    y_pred = np.argmax(y, axis=1)  
    accuracy = np.mean(y_pred == t) 
    return accuracy
```
* 通过 `argmax` 找到每个样本的预测类别
* `accuracy = np.mean(y_pred == t) `计算平均精确度

#### 3.5.5 梯度计算
```
def gradient(self, x, t):
    grads = {}
    # 定义损失函数
    loss_fn = self.loss
    # 计算梯度
    grads['W1'] = numerical_gradient(lambda W1: loss_fn(x, t), self.params['W1'])
    grads['b1'] = numerical_gradient(lambda b1: loss_fn(x, t), self.params['b1'])
    grads['W2'] = numerical_gradient(lambda W2: loss_fn(x, t), self.params['W2'])
    grads['b2'] = numerical_gradient(lambda b2: loss_fn(x, t), self.params['b2'])

    return grads
```
* `grads['W1'] = numerical_gradient(lambda W1: loss_fn(x, t), self.params['W1'])`用W1返回 loss_fn(x, t)

### 3.6 模型训练
#### 3.6.1 参数定义
```
iters_num = 10000 
train_size = x_train.shape[0]  
batch_size = 100  
learning_rate = 0.1

network = TwoLayerNet(input_size, hidden_size, output_size)
```

#### 3.6.2 计算epoch的迭代次数
```
iter_per_epoch = max(train_size / batch_size, 1)
```

#### 3.6.3 训练循环
```
for i in range(iters_num):
    # 在每次训练迭代内部选择一个批次的数据
    batch_mask = np.random.choice(train_size, batch_size)  # 随机选择批次
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # print(f"x_batch.size = {x_batch.size}")

    grads = network.gradient(x_batch, t_batch)

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

### 3.7 结果展示
* 运算非常非常非常慢，大半天才跑一轮

<img width="569" alt="{8E5D8C55-9B6A-4269-939A-D56E74EAFD5C}" src="https://github.com/user-attachments/assets/cd11002e-f81e-4371-b9ee-006c29f90dec">

## 四、整体程序
```
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [0, 1]
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

x_full_train = full_train_dataset.data.numpy()  # 原始图像数据，shape = (60000, 28, 28)
t_full_train = full_train_dataset.targets.numpy()  # 原始标签数据，shape = (60000,)

x_full_train = x_full_train.reshape(-1, 784)

# 按照 82% 和 18% 的比例拆分训练集数据
x_train, x_test, t_train, t_test = train_test_split(x_full_train, t_full_train, test_size=0.18, random_state=42)
```

```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=-1, keepdims=True)

def cross_entropy_error(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    if y_true.ndim == 1:
        y_true = np.eye(y_pred.shape[1])[y_true]
    return - (y_true * np.log(y_pred)).sum(axis=1).mean()

def numerical_gradient(f, x):
    epsilon = 1e-7
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + epsilon
        f_plus = f(x)
        x[idx] = tmp_val - epsilon
        f_minus = f(x)
        grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        x[idx] = tmp_val
        it.iternext()
    return grad
```

```
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, x):
        # 获取网络参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 第一层的运算
        z1 = np.dot(x, W1) + b1
        h1 = sigmoid(z1)
        z2 = np.dot(h1, W2) + b2
        return softmax(z2)

    def loss(self, x, t):
        y = self.forward(x)
        return cross_entropy_error(t, y)

    def accuracy(self, x, t):
        y = self.forward(x)
        y_pred = np.argmax(y, axis=1)
        accuracy = np.mean(y_pred == t)
        return accuracy

    def gradient(self, x, t):
        grads = {}
        
        # 定义损失函数
        loss_fn = self.loss
        
        # 计算梯度
        grads['W1'] = numerical_gradient(lambda W1: loss_fn(x, t), self.params['W1'])
        grads['b1'] = numerical_gradient(lambda b1: loss_fn(x, t), self.params['b1'])
        grads['W2'] = numerical_gradient(lambda W2: loss_fn(x, t), self.params['W2'])
        grads['b2'] = numerical_gradient(lambda b2: loss_fn(x, t), self.params['b2'])

        return grads
```

```
iters_num = 10000  
train_size = x_train.shape[0]
print(f"train_size:{train_size}")
batch_size = 100
learning_rate = 0.1

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
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)

        test_acc = network.accuracy(x_test, t_test)
        test_acc_list.append(test_acc)

        print(f"iteration:{i}, train acc: {train_acc}, test acc: {test_acc}, loss: {loss}")
```

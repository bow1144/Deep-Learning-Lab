## 一、观察SGD算法的更新过程

### 1.1 定义SGD网络
```
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def step(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

### 1.2 观察网络的更新
```
import matplotlib.pyplot as plt
from collections import OrderedDict

def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

optimizer = SGD(lr=0.95)
x_history = []
y_history = []
params['x'], params['y'] = init_pos[0], init_pos[1]
    
for i in range(30):
    x_history.append(params['x'])
    y_history.append(params['y'])
        
    grads['x'], grads['y'] = df(params['x'], params['y'])
    optimizer.step(params, grads)
    

x = np.arange(-10, 10, 0.01)
y = np.arange(-5, 5, 0.01)
    
X, Y = np.meshgrid(x, y) 
Z = f(X, Y)
    
mask = Z > 7
Z[mask] = 0
    
# plot 
plt.plot(x_history, y_history, 'o-', color="red")
plt.contour(X, Y, Z)
plt.ylim(-10, 10)
plt.xlim(-10, 10)
plt.plot(0, 0, '+')

plt.title('image')
plt.xlabel("x")
plt.ylabel("y")
    
plt.show()
```

* 运行结果：

<img width="452" alt="{B2797FBF-62B2-4123-9E01-7BBAAA4A6754}" src="https://github.com/user-attachments/assets/824d1279-6a9b-4b28-b7db-b4c78a9026d5" />

### 1.3 Momentum算法观察SGD的更新过程
```
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, params, grads):
        # 初始化速度 v
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)

        # Momentum 参数更新算法
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]  # 更新速度
            params[key] += self.v[key]  # 更新参数
```

<img width="460" alt="{8360CFB4-70CA-42BC-9885-3E3695994D2D}" src="https://github.com/user-attachments/assets/da3f7564-9765-4d97-8d43-d8a74afb655f" />

### 1.4 AdaGrad算法算法观察SGD的更新过程
```
class AdaGrad:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def step(self, params, grads):
        # 初始化内部变量 h
        if self.h is None:
            self.h = {key: np.zeros_like(value) for key, value in params.items()}

        # AdaGrad 参数更新算法
        for key in params.keys():
            self.h[key] += grads[key] ** 2  # 累积梯度平方
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)  # 参数更新
```

<img width="457" alt="{81604856-5021-43CB-961C-00041E929C96}" src="https://github.com/user-attachments/assets/3365eadf-f2c5-432e-8e04-9e00b0525b26" />

## 二、权重的初始化

### 2.1 随机生成的数据
```
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

input_data = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 改变初始值进行实验！
    w = np.random.randn(node_num, node_num) * 1   #标准差为1的正态分布

    a = np.dot(x, w)

    # 将激活函数的种类也改变，来进行实验！
    z = sigmoid(a)

    activations[i] = z

# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
```

<img width="425" alt="{02D559A9-B54D-4FF4-B845-BACF7C218463}" src="https://github.com/user-attachments/assets/a4ec5ebb-d1d1-4988-9323-9abaa8d1c478" />

### 2.2 将标准差设置为0.01

```
w = np.random.randn(node_num, node_num) * 0.01
```

<img width="425" alt="{065E4CFF-5EA0-4F68-B50B-6D0F8EC533EF}" src="https://github.com/user-attachments/assets/de509a38-0e9a-4358-8bdd-58d854651bbb" />

### 2.3 Xavier初始值

```
w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
```

<img width="420" alt="{AB416554-3363-4ED7-A83D-AFA2DFD35642}" src="https://github.com/user-attachments/assets/f30cab14-21ec-4371-8434-72ee54cd191f" />

### 2.4 He初始值

```
w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
```

<img width="459" alt="{1B8E1A42-1CA4-458A-81C7-E05A74D646AA}" src="https://github.com/user-attachments/assets/7b6b9bec-d44c-4e26-9a9e-6f3d29d7f5e4" />

## 三、Batch Normalization

### 3.1 加载数据集
```
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
epochs = 10
```

### 3.2 Batch Normalization
```
class BatchNormalization:

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        #将每一个样本由三维数组转换为一维数组
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        #在__forward方法中完成BatchNorm函数的前向传播
        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
```

### 3.3 普通神经网络
```
class SimpleNN(nn.Module):
    def __init__(self, use_batch_norm=False):
        super(SimpleNN, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256) if use_batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
```

### 3.4 运行结果

<img width="474" alt="{53C40BD3-8E3F-46FA-9B72-11E5F1DB76E4}" src="https://github.com/user-attachments/assets/5df231c9-bee0-43ac-98a1-2c5f02c7ba93" />

<img width="228" alt="{A38F3F46-CF1F-4311-A1C7-8971155BA85E}" src="https://github.com/user-attachments/assets/15a42a82-ae28-498f-8a6e-640f1c1b51a7" />



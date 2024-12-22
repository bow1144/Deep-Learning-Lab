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

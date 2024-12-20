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

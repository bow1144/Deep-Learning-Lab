# 作业一、实验环境搭建与典型应用认知

## 一、环境搭建
由于设备在之前已经配置`pyotorch`,`Anaconda`与`cudnn-cpu`，故跳过环境搭建。
![环境](image/conda环境.png)

## 二、Mnist数据集载入

* 从`mnist_dataset`数据集导入，并显示其中一个数字  
![2](image/2.png)

* 从矩阵中观察数据集特征  
![3](image/3.png)

## 三、用基础参数训练  
![4](image/4.png)

* 训练过程  
![5](image/5.png)

* 查看损失函数  
![6](image/6.png)

## 四、测试  
![7](image/7.png)

## 五、GAN网络训练人脸数据
### 5.1 建立训练模型
![8](image/8.png)

### 5.2 开始训练
> 由于本机没有GPU，故为了训练效果，使用云端GPU `T4`，总训练时长约`100min`

![9](image/9.png)

### 5.3 训练结果

![10](image/0_1000.png)

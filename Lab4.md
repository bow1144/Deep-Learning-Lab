# 目标检测

## 一、准备工作

### 1.1 下载数据集
* 直接在网站下载，并将`.zip`文件作为数据

### 1.2 数据集预览
```
from PIL import Image

img_files = os.listdir(train_img_path)

img_files = img_files[:10]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 创建2行5列的子图
axes = axes.flatten()  # 将axes扁平化，方便索引

# 逐个显示图片
for i, img_file in enumerate(img_files):
    img_path = os.path.join(train_img_path, img_file)
    img = Image.open(img_path)
    
    axes[i].imshow(img)
    axes[i].axis('off')  # 关闭坐标轴显示
    axes[i].set_title(f'Image {i+1}')

plt.tight_layout()  # 自动调整子图的布局
plt.show()
```

![{AB22FF28-4DAB-4D38-AFE8-1C2F7E66BEA3}](https://github.com/user-attachments/assets/bf1f36c8-6ece-447a-8345-86182c7ebe9b)

### 1.3 类别预测层

```
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

这个函数返回一个卷积层，它的输入通道数是 `num_inputs`，
输出通道数是 `num_anchors * (num_classes + 1)`，即对于每个锚框，
要预测 (num_classes + 1) 个输出，其中 num_classes 是类别数，+1 是背景类别的预测。

### 1.4 边界框预测层
```
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

返回一个卷积层，每个锚框预测四个边界坐标

### 1.5 连结多尺度的预测

```
def forward(x, block):
    return block(x)
```
接收张量和网络块，向前传播

```
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

* 创建网络快，输出通道数为 $ 5 * (10 + 1) $
* 对于第一个，输出通道数为55，尺寸数是20

![{A257369A-B129-4E3C-A8A1-9D63FA7308E0}](https://github.com/user-attachments/assets/d9ad6385-bcae-4211-9c36-6eda8090e3d6)

```
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)
```
* `pred.permute(0, 2, 3, 1)`的作用是维度重排
* ` start_dim=1`从第一维开始全部展平

```
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```
* 将多个预测结果展开并拼接
* 以Y1为例，先维度交换为`(2, 20, 20, 55)`，再展开后变成`(2, 22000)`



![{5C1635A0-E131-416B-9789-C28723BEE4A8}](https://github.com/user-attachments/assets/4de26157-b146-4404-b2ca-7e58df5f5f41)


### 1.7 高宽减半
```
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```
* 定义了卷积层、归一化层、激活层和最大池化层
* 两个卷积层

```
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```
* 输入通道为3，输出通道为10
* 输出张量：第二维度的3变为10，池化步为2
* 最终输出为`(2, 10, 10, 10)`


![{DB013403-FCD9-4A11-BA6B-6DB8777691EF}](https://github.com/user-attachments/assets/95b599fc-c391-4cbb-9e66-c5be7eb1c866)

### 1.8 基本网络块
```
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```
* 定义了一个基本网络

### 1.9 完整的模型
```
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
num_anchors
```
* i=0时返回包含多个卷积池的池化层网络
* i=1时返回`down_sample_blk(64, 128)`，表示从 64 通道到 128 通道的下采样模块
* i=4返回 `nn.AdaptiveMaxPool2d((1, 1))`，表示将输出特征图大小调整为 (1, 1)
* `num_anchors` 通过公式 `len(sizes[0]) + len(ratios[0]) - 1` 计算得到

## 二、实现multibox_priori和multibox_target函数
```
import torch

def multibox_prior(feature_map, sizes, ratios):

    height, width = feature_map.shape
    num_anchors = len(sizes) + len(ratios) - 1  # 第一种 size 与所有 ratio 结合
    
    anchors = []
    for i in range(height):
        for j in range(width):
            center_y = (i + 0.5) / height
            center_x = (j + 0.5) / width
            
            for size in sizes:
                for ratio in ratios:
                    w = size * (ratio ** 0.5) / width
                    h = size / (ratio ** 0.5) / height
                    anchors.append([center_x, center_y, w, h])
    
    return torch.tensor(anchors).view(height * width * num_anchors, 4)
```
* 输入的特征图，是卷积网络层的输出，size是候选框的大小，ratios用来定义候选框的长宽比
* `num_anchors = len(sizes) + len(ratios) - 1`计算锚框的数量
* 生成锚框并计算宽高
* `return torch.tensor(anchors).view(height * width * num_anchors, 4)`返回锚框

![{065ACB60-0EAB-40F0-AFD6-AB72182C8BA2}](https://github.com/user-attachments/assets/84390e48-79f8-4d67-a073-a5450cc990d4)

```
def multibox_target(anchors, gt_boxes, iou_threshold=0.5):

    def calc_iou(box1, box2):
        inter_xmin = torch.max(box1[:, 0], box2[:, 0])
        inter_ymin = torch.max(box1[:, 1], box2[:, 1])
        inter_xmax = torch.min(box1[:, 2], box2[:, 2])
        inter_ymax = torch.min(box1[:, 3], box2[:, 3])
        
        inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area.clamp(min=1e-6)

    num_anchors = anchors.size(0)
    num_gt = gt_boxes.size(0)
    
    # 转换 anchors 为 (xmin, ymin, xmax, ymax) 格式
    anchors_corners = torch.cat([
        anchors[:, :2] - anchors[:, 2:] / 2,  # xmin, ymin
        anchors[:, :2] + anchors[:, 2:] / 2   # xmax, ymax
    ], dim=-1)
    
    # 计算 IOU
    iou_matrix = torch.zeros((num_anchors, num_gt))
    for i in range(num_gt):
        iou_matrix[:, i] = calc_iou(anchors_corners, gt_boxes[i].unsqueeze(0))
    
    # 为每个锚框分配标签
    labels = torch.zeros(num_anchors, dtype=torch.long)
    offsets = torch.zeros_like(anchors)
    
    for i in range(num_anchors):
        max_iou, max_idx = iou_matrix[i].max(dim=0)
        if max_iou >= iou_threshold:
            gt_box = gt_boxes[max_idx]
            labels[i] = 1  # 假设类别为 1
            dx = (gt_box[0] - anchors[i, 0]) / anchors[i, 2]
            dy = (gt_box[1] - anchors[i, 1]) / anchors[i, 3]
            dw = torch.log(gt_box[2] / anchors[i, 2])
            dh = torch.log(gt_box[3] / anchors[i, 3])
            offsets[i] = torch.tensor([dx, dy, dw, dh])
    
    return labels, offsets
```

* `multibox_target` 函数的目的是根据真实的目标框和锚框之间的匹配关系，
  生成训练所需的标签和偏移量
* 计算 IOU 函数：计算两个匡的交并比
* `iou_matrix `计算锚框和目标框的 IOU 矩阵，用于存储每个锚框和每个真实框之间的 IOU 值
* 为每个框分配标签和偏移量
* 最终返回每个锚框的标签 labels 和偏移量 offsets，
  labels 用于训练时进行分类，offsets 用于训练时进行边界框回归

## 三、训练和测试SSD模型
```
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```
* 动态创建网络层
  * get_blk(i): 该函数返回一个卷积块，目的是减少特征图的空间尺寸
  * cls_predictor: 用于生成分类预测的卷积层
  * bbox_predictor: 用于生成边界框回归的卷积层
* forward方法：输入一个图像，并通过网络的卷积层生成锚框、分类预测和边界框回归预测
  * blk_forward 将图像输入卷积层，返回特征图，分类预测和边框回归预测
  * 将每一层的预测合并
  * 返回一个包含框的张量、一个分类张量、回归预测张量

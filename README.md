# PyTorch-ImageClassifier

数字图像课设，图片分类  
选择垃圾分类

---

## 数据集选择

来自*Kaggle*的垃圾分类数据集：[下载地址](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/code)  
一共有六类垃圾:纸板（cardboard）、玻璃（glass）、金属（metal）、纸（paper）、塑料（plastic）、其他垃圾（trash）  
393（cardboard） + 491（glass） + 400（metal） + 584（paper） + 472（plastic） + 127（trash） = 2467 张  
403 501 410 594 482 137 = 2527

Garbage Classification Data  
The Garbage Classification Dataset contains 6 classifications:

1. cardboard (393)
2. glass (491)
3. metal (400)
4. paper(584)
5. plastic (472)
6. trash(127).

### 垃圾分类（12类）

另一个数据集：[下载地址](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

ImageFolder类还会自动为每个图像分配一个标签。标签是基于图像文件夹的子文件夹的名称分配的。

```

/Garbage classification/

cardboard/
cardboard1.jpg
cardboard2.jpg

glass/
glass1.jpg
glass2.jpg
```

则ImageFolder将为cardboard文件夹中的所有图像分配标签0，将为glass文件夹中的所有图像分配标签1

```
import torchvision.transforms as transforms  # transforms 模块包含用于图像预处理的各种转换操作
from torchvision.datasets import ImageFolder  # ImageFolder 类是用于加载具有类别标签的图像数据集的便捷工具

ImageFolder(data_dir, transform=transformers['demo1']),
```

```markdown
transformers = {
'demo1': transforms.Compose([
transforms.Resize((256, 256)),
transforms.ToTensor()
]),
'demo2': transforms.Compose([
transforms.Resize((256, 256)),

# 随机图像亮度、对比度、饱和度

transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

# 随机翻转

# transforms.RandomHorizontalFlip(),

transforms.RandomRotation(5),

# 随机放射变化

transforms.RandomAffine(degrees=11, translate=(0.1, 0.1), scale=(0.8, 0.8)),
transforms.ToTensor(),

# 标准化图像，使用 ImageNet 的均值和标准差

transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化
]),
}
```

### 猫狗

数据集：[下载地址](https://www.kaggle.com/datasets/lizhensheng/-2000/discussion)

## 问题

tmd我是说我训练的模型怎么效果这么拉跨，后面在kaggle训练的高分模型不如之前的，我才发现，kaggle上的标签读取它的顺序不一样
而我那种映射是错误的，就是用class_index 前面没发现问题，是因为刚好和正确的顺序一样

```markdown
# ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

class_index = ['纸板', '玻璃', '金属', '纸', '塑料', '其他垃圾']

# ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']

class_index2 = ['金属', '玻璃', '纸', '其他垃圾', '纸板', '塑料']
```

```python
def predict_img(path, model):
    device = get_default_device()
    image = Image.open(path)
    # 大小调整为256x256像素，并且转换为Pytorch张量
    image_clean = transformers['demo4'](image)
    # 将图像转换为张量并移动到device
    xb = to_device(image_clean.unsqueeze(0), device)
    # 预测
    yb = model(xb)
    # 选择概率最高的类别
    prob, preds = torch.max(yb, dim=1)
    # print(prob.item())
    index = preds[0].item()
    print("index", index)
    label = dataset.classes[index]
    print("label", label)
    print(dataset.classes)
    # show_pred_image(image, index, label)
    return label, prob.item()
```
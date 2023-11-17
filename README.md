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
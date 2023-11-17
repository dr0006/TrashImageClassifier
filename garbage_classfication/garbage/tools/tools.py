# -*- coding: utf-8 -*-
"""
@File  : tools.py
@author: FxDr
@Time  : 2023/11/12 21:28
@Description:
"""
import matplotlib.pyplot as plt

from garbage_classfication.garbage.config import dataset

# 中文绘制
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 定义类别名称的中文对应关系
class_mapping = {
    "cardboard": "纸板",
    "glass": "玻璃",
    "metal": "金属",
    "paper": "纸",
    "plastic": "塑料",
    "trash": "其他垃圾",
}

class_index = ['纸板', '玻璃', '金属', '纸', '塑料', '其他垃圾']


def show_sample(img, label):
    """绘制数据集的个别样例来查看一下"""
    class_name = class_mapping[dataset.classes[label]]
    print("标签:", class_name, "(类别编号：" + str(label) + ")")

    # 使用 PyTorch 的 permute 函数交换通道顺序，以适应 matplotlib 的图像显示格式
    plt.imshow(img.permute(1, 2, 0))
    plt.title(class_name)
    plt.show()


def plot_accuracies(history):
    """绘制准确率曲线"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


def plot_losses(history):
    """绘制损失曲线"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

# img, label = dataset[100]
# show_sample(img, label)

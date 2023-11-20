# -*- coding: utf-8 -*-
"""
@File  : predict_img.py
@author: FxDr
@Time  : 2023/11/12 22:55
@Description:
"""
import torch
from PIL import Image
from matplotlib import pyplot as plt

import torchvision.transforms as transforms  # transforms 模块包含用于图像预处理的各种转换操作

from garbage_classfication.garbage.config import dataset
from garbage_classfication.garbage.model import to_device, get_default_device
from garbage_classfication.garbage.tools.tools import class_index

transformations = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def predict_img(path, model):
    device = get_default_device()
    image = Image.open(path)
    # 大小调整为256x256像素，并且转换为Pytorch张量
    image_clean = transformations(image)
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
    label = class_index[index]
    # print(dataset.classes)
    # show_pred_image(image, index, label)
    return label, prob.item()


def show_pred_image(image, index, label):
    """绘制预测的图片带标签"""
    # print("标签:", label, "(类别编号：" + str(index) + ")")
    plt.imshow(image)
    plt.title(label)
    plt.show()

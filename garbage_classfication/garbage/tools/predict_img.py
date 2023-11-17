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

from garbage_classfication.garbage.config import transformers
from garbage_classfication.garbage.model import to_device, get_default_device
from garbage_classfication.garbage.tools.tools import class_index


def predict_img(path, model):
    device = get_default_device()
    image = Image.open(path)
    # 大小调整为256x256像素，并且转换为Pytorch张量
    image_clean = transformers['demo1'](image)
    # 将图像转换为张量并移动到device
    xb = to_device(image_clean.unsqueeze(0), device)
    # 预测
    yb = model(xb)
    # 选择概率最高的类别
    prob, preds = torch.max(yb, dim=1)
    # print(prob.item())
    index = preds.item()
    label = class_index[index]
    # show_pred_image(image, index, label)
    return label, prob.item()


def show_pred_image(image, index, label):
    """绘制预测的图片带标签"""
    # print("标签:", label, "(类别编号：" + str(index) + ")")
    plt.imshow(image)
    plt.title(label)
    plt.show()

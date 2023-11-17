# -*- coding: utf-8 -*-
"""
@File  : test1.py
@author: FxDr
@Time  : 2023/11/12 21:21
@Description:
"""
import torch
from torch.utils.data import random_split

from garbage_classfication.garbage.config import dataset, split_sizes
from garbage_classfication.garbage.model import to_device, get_default_device, ResNet
from garbage_classfication.garbage.tools.tools import show_sample, class_mapping


def predict_image(img, model):
    # 将图像转换为张量并移动到device
    xb = to_device(img.unsqueeze(0), device)
    # 预测
    yb = model(xb)
    # 选择概率最高的类别
    prob, preds = torch.max(yb, dim=1)
    return dataset.classes[preds[0].item()]


train_ds, val_ds, test_ds = random_split(dataset, split_sizes)
device = get_default_device()
print("Using device:{}".format(device))

model = to_device(ResNet(), device)
model.load_model_dict('../model/resnet50/95.54%_model_weights.pth')

img, label = test_ds[17]
show_sample(img, label)
print('类别:', class_mapping[dataset.classes[label]], ', 预测:', class_mapping[predict_image(img, model)])

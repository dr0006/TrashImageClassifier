# -*- coding: utf-8 -*-
"""
@File  : test2.py
@author: FxDr
@Time  : 2023/11/12 23:05
@Description:
"""
from garbage_classfication.garbage.tools.predict_img import predict_img
from garbage_classfication.garbage.model import get_default_device, to_device, ResNet

device = get_default_device()
print("Using device:{}".format(device))
model = to_device(ResNet(), device)
# model.load_model_dict('model/resnet50/95.54%_model_weights.pth')
model.load_model_dict('../model/resnet50/94.50%_model_weights.pth')
path = "../../test_images/garbage/plastic.jpeg"
# path = "../../test_images/glass.jpg"
# path = "../../test_images/paper.jpg"
label = predict_img(path, model)
print("预测类别为:{}".format(label))

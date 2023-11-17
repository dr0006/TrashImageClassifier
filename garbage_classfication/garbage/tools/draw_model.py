# -*- coding: utf-8 -*-
"""
@File  : draw_model.py
@author: FxDr
@Time  : 2023/11/13 0:01
@Description:
"""
import netron

# 启动 Netron 可视化
# netron.start('./model/resnet50/onnx/95.54%garbage_class.onnx')
netron.start(
    r'/garbage_classfication/model/resnet50/onnx/95.54%garbage_class.onnx')

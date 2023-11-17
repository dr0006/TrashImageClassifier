# -*- coding: utf-8 -*-
"""
@File  : onnx_model.py
@author: FxDr
@Time  : 2023/11/13 0:12
@Description:
"""
import torch

from garbage_classfication.garbage.config import example_input
from garbage_classfication.garbage.model import ResNet

model = ResNet()

# 将模型设置为评估模式
model.eval()

# 将模型的权重加载到设备上
model.load_state_dict(torch.load('../../model/resnet50/95.54%_model_weights.pth'))
# 将模型加载到设备上
model.to(example_input.device)

# 导出模型到 ONNX 格式
onnx_file_path = '../../model/resnet50/onnx/95.54%garbage_class.onnx'
torch.onnx.export(model, example_input, onnx_file_path, verbose=True)

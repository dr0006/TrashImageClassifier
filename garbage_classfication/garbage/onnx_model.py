# -*- coding: utf-8 -*-
"""
@File  : onnx_model.py
@author: FxDr
@Time  : 2023/11/13 0:12
@Description:
"""
import torch

from garbage_classfication.garbage.model import ImageClassificationBase

# from garbage_classfication.garbage.model import ResNet

# from garbage_classfication.garbage.config import example_input
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet50_Weights


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.best_accuracy = 0.0
        self.best_model_state = None  # 最高测试集评分的参数副本
        # 使用预训练模型 ResNet-50
        self.network = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 替换最后一层全连接层
        num_ftr_s = self.network.fc.in_features
        # dataset2 demo2
        self.network.fc = nn.Linear(num_ftr_s, 12)

    def forward(self, xb):
        # 使用 sigmoid 函数处理输出
        return torch.sigmoid(self.network(xb))

    def save_model(self):
        """
        保存模型
        """
        # 转换百分比形式
        accuracy_percent = f'{self.best_accuracy * 100:.2f}%'
        # 保存模型的权重
        # torch.save(self.state_dict(), 'model/model_weights.pth')
        # 整个模型
        # torch.save(self, "model/entire_model.pth")
        # 保存最佳模型参数和整个模型
        best_model_weights_filename = f'model/{accuracy_percent}_model_weights.pth'
        best_model_filename = f'model/{accuracy_percent}_entire_model.pth'
        torch.save(self.best_model_state, best_model_weights_filename)
        torch.save(self, best_model_filename)

    def load_model_dict(self, path):
        """
        加载模型的权重
        :param path: 路径
        """
        self.load_state_dict(torch.load(path))
        self.eval()  # 设置为评估模式


model = ResNet()

# 将模型设置为评估模式
model.eval()

batch_size = 25  # 定义批量大小
example_input = torch.randn(batch_size, 3, 256, 256)

# 将模型的权重加载到设备上
model.load_state_dict(torch.load(
    r'X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\model\model_files12\97.62'
    r'%_ResNet50_model_weights.pth'))
# 将模型加载到设备上
model.to(example_input.device)

# 导出模型到 ONNX 格式
onnx_file_path = r'X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\model\model_files12\97.62%_ResNet50' \
                 r'.onnx'
torch.onnx.export(model, example_input, onnx_file_path, verbose=True)

# -*- coding: utf-8 -*-
"""
@File  : predict_model.py
@author: FxDr
@Time  : 2023/11/20 23:09
@Description:
"""
import torch
from PIL import Image
from torchvision import transforms, models

import warnings

warnings.filterwarnings("ignore")


class ImageClassifier:
    def __init__(self, svm_model_path, pca_model_path):
        # 加载保存的SVM模型
        self.clf = torch.load(svm_model_path)

        # 使用保存的PCA模型
        self.pca = torch.load(pca_model_path)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # ResNet模型提取特征
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

    def predict_image(self, new_image_path):
        # 读取新图像
        image = Image.open(new_image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)

        # 提取特征
        with torch.no_grad():
            outputs = self.model(image)

        features = outputs.squeeze().numpy()

        # 使用PCA降维
        features_pca = self.pca.transform(features.reshape(1, -1))

        # 预测
        predicted_label = self.clf.predict(features_pca)[0]

        # 映射预测标签到类别名称
        predicted_class = 'cat' if predicted_label == 0 else 'dog'

        return predicted_class




# -*- coding: utf-8 -*-
"""
@File  : test.py
@author: FxDr
@Time  : 2023/11/20 22:53
@Description:
"""
import torch
from PIL import Image
from torchvision import transforms, models

import warnings

warnings.filterwarnings("ignore", )

# 加载保存的SVM模型
saved_model_path = '../pth/svm_model.pth'
clf = torch.load(saved_model_path)

# 读取新图像
new_image_path = r"C:\Users\lenovo\Downloads\archive\cat和dog\test\dog\dog.1487.jpg"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(new_image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# 提取特征
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

with torch.no_grad():
    outputs = model(image)

features = outputs.squeeze().numpy()

# 使用PCA降维
n_components = 150
# 使用保存的PCA模型降维
pca = torch.load('../pth/pca_model.pth')
features_pca = pca.transform(features.reshape(1, -1))

# 预测
predicted_label = clf.predict(features_pca)[0]

# 映射预测标签到类别名称
predicted_class = 'cat' if predicted_label == 0 else 'dog'

print(f'新图像的预测类别是：{predicted_class}')

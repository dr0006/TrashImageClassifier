# -*- coding: utf-8 -*-
"""
@File  : test.py
@author: FxDr
@Time  : 2023/11/21 0:18
@Description:
"""
import joblib
import torch
from PIL import Image
from torchvision import transforms, models
import warnings

warnings.filterwarnings('ignore')

# 加载K均值聚类模型
kmeans_loaded = joblib.load('./joblib_files/kmeans_model.joblib')

# 加载PCA模型
pca_loaded = joblib.load('./joblib_files/pca_model.joblib')

# 加载预训练的ResNet模型
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image_path, model, pca, kmeans, label_mapping):
    """
    预测新图像的类别
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # 提取特征
    model.eval()
    with torch.no_grad():
        outputs = model(image)

    features = outputs.squeeze().numpy()
    # 使用PCA降维
    features_pca = pca.transform(features.reshape(1, -1))
    # 预测
    predicted_label = kmeans.predict(features_pca)[0]
    # 映射预测标签到类别名称
    predicted_class = label_mapping[predicted_label]

    return predicted_class


# 预测标签到类别名称的映射
label_mapping = {0: 'paper', 1: 'Metal', 2: 'clothes', 3: 'shoes'}  # 替换为实际类别名称
# 测试一个新图像
new_image_path = r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\paper\paper166.jpg"
predicted_class = predict_image(new_image_path, model, pca_loaded, kmeans_loaded, label_mapping)

print(f'新图像的预测类别是：{predicted_class}')

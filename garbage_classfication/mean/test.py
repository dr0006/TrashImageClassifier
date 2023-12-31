# -*- coding: utf-8 -*-
"""
@File  : test.py
@author: FxDr
@Time  : 2023/11/21 0:18
@Description:
"""
import os
import joblib
import torch
from PIL import Image
from torchvision import transforms, models

import time

import warnings

warnings.filterwarnings('ignore')

# 记录开始时间
start_time = time.time()

# 加载K均值聚类模型
print('kmeans Loading....')
kmeans_loaded = joblib.load('./joblib_files/kmeans_model.joblib')

# 记录结束时间
end_time = time.time()

# 打印加载时间
print(f'K-means model loaded in {end_time - start_time} seconds')

# 记录开始时间
start_time = time.time()

# 加载PCA模型
print('PCA Loading....')
pca_loaded = joblib.load('./joblib_files/pca_model.joblib')

# 记录结束时间
end_time = time.time()

# 打印加载时间
print(f'PCA model loaded in {end_time - start_time} seconds')

# 加载预训练的ResNet模型
print("model Loading....")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 去掉了最后一层，用于特征提取
# print(model)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image_path, model, pca, kmeans, label_mapping):
    """
    预测新图像的类别
    """
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


def batch_predict_images(folder_path, model, pca, kmeans, label_mapping):
    predictions = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            predicted_class = predict_image(image_path, model, pca, kmeans, label_mapping)
            predictions.append((filename, predicted_class))

    return predictions


# 预测标签到类别名称的映射
label_mapping = {0: 'paper', 1: 'Metal', 2: 'clothes', 3: 'shoes'}  # 替换为实际类别名称
"""
# 测试一个文件夹中的所有图像
folder_path = r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\bb"
batch_predictions = batch_predict_images(folder_path, model, pca_loaded, kmeans_loaded, label_mapping)

for filename, predicted_class in batch_predictions:
    print(f'{filename}: 预测类别是 {predicted_class}')

"""

# img_path = r"X:\Coding\Github\PyTorch-ImageClassifier\test_images\shoe.png"
# img_path = r"X:\Coding\Github\PyTorch-ImageClassifier\test_images\shoe1.png"
img_path = r"X:\Coding\Github\PyTorch-ImageClassifier\test_images\clothes.png"
pre_label = predict_image(img_path, model=model, pca=pca_loaded, kmeans=kmeans_loaded, label_mapping=label_mapping)
print(pre_label)

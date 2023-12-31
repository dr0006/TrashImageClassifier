# -*- coding: utf-8 -*-
"""
@File  : test2.py
@author: FxDr
@Time  : 2023/12/26 19:49
@Description:
"""
import os
import re
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


def extract_true_label(filename):
    """
    从文件名中提取真实标签
    """
    match = re.match(r'([a-zA-Z]+)_?\d*\.\w+', filename)
    if match:
        return match.group(1).lower()
    else:
        return None


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


def batch_predict_images(folder_paths, model, pca, kmeans, label_mapping):
    for folder_path in folder_paths:
        print(f"\nTesting images in folder: {folder_path}")

        correct_predictions = 0
        total_images = 0

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
                image_path = os.path.join(folder_path, filename)

                # 提取真实标签
                true_class = extract_true_label(filename)
                if true_class is None:
                    print(f'无法提取真实标签，跳过文件：{filename}')
                    continue

                predicted_class = predict_image(image_path, model, pca, kmeans, label_mapping)

                if predicted_class == true_class:
                    correct_predictions += 1

                print(f'{filename}: 真实类别是 {true_class}, 预测类别是 {predicted_class}')

        accuracy = correct_predictions / total_images if total_images > 0 else 0
        print(
            f'\n总共测试了 {total_images} 张图像，预测正确的数量为 {correct_predictions}，准确率为 {accuracy * 100:.2f}%')


# 预测标签到类别名称的映射
label_mapping = {0: 'paper', 1: 'metal', 2: 'clothes', 3: 'shoes'}  # 替换为实际类别名称

# 测试多个文件夹中的所有图像
folder_paths = [
    r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\clothes",
    r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\paper",
    r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\Metal",
    r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\shoes"
]

batch_predict_images(folder_paths, model, pca_loaded, kmeans_loaded, label_mapping)

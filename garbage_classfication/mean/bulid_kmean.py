# -*- coding: utf-8 -*-
"""
@File  : bulid_kmean.py
@author: FxDr
@Time  : 2023/11/19 17:55
@Description:
"""
import os
import warnings

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import joblib_files

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Microsoft YaHei'


def extract_features(model, dataloader):
    """
    使用预训练模型提取图像特征
    """
    print("使用预训练模型提取图像特征")
    model.eval()
    features = []
    with torch.no_grad():
        for images in dataloader:
            outputs = model(images)
            features.extend(outputs.squeeze().numpy())
    return torch.tensor(features)


def apply_pca(data, n_components=200):
    """
    使用PCA进行降维
    """
    print("使用PCA进行降维")
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca, pca


def perform_clustering(data, num_clusters=4):
    """
    使用K均值聚类
    """
    print("使用K均值聚类")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(data)
    return predicted_labels, kmeans


def visualize_clusters(data, predicted_labels, num_clusters=4):
    """
    可视化聚类结果
    """
    print("可视化聚类结果")
    plt.figure(figsize=(10, 6))
    for i in range(num_clusters):
        cluster_points = data[predicted_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

    plt.title('无监督图像聚类')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.legend()
    plt.show()


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


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


# 设置数据集路径
dataset_path = '../bb'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载图像
dataset = CustomDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 使用预训练的ResNet模型提取特征
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# 提取特征
features = extract_features(model, dataloader)

# 使用PCA降维
features_pca, pca = apply_pca(features)

# 使用K均值聚类
num_clusters = 4  # 聚类数
predicted_labels, kmeans = perform_clustering(features_pca, num_clusters)

# 可视化聚类结果
visualize_clusters(features_pca, predicted_labels, num_clusters)

# 预测标签到类别名称的映射
label_mapping = {0: '类别1', 1: '类别2', 2: '类别3', 3: '类别4'}  # 替换为实际类别名称


def count_images_in_folder(test_folder, model, pca, kmeans, label_mapping):
    """
    统计一个文件夹下各个类别的图像数量
    """
    class_count = {label: 0 for label in label_mapping.values()}

    # 遍历文件夹下的所有图片文件
    for filename in os.listdir(test_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 构建完整的图像路径
            image_path = os.path.join(test_folder, filename)

            # 预测图像类别
            predicted_class = predict_image(image_path, model, pca, kmeans, label_mapping)

            # 更新类别计数
            class_count[predicted_class] += 1

    # 打印各个类别的数量
    print(f"文件夹 {test_folder} 中各个类别的数量:")
    for label, count in class_count.items():
        print(f"{label}: {count}")


# 测试多个路径
test_folders = [
    r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\Metal",
    r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\clothes",
    r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\paper",
    r"X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\mean\test_data\shoes"
]

for folder in test_folders:
    count_images_in_folder(folder, model, pca, kmeans, label_mapping)

# 保存K均值聚类模型
kmeans_model_path = 'joblib_files/kmeans_model.joblib'
joblib_files.dump(kmeans, kmeans_model_path)
print(f'K均值聚类模型已保存到: {kmeans_model_path}')

# 保存PCA模型
pca_model_path = 'joblib_files/pca_model.joblib'
joblib_files.dump(pca, pca_model_path)
print(f'PCA模型已保存到: {pca_model_path}')

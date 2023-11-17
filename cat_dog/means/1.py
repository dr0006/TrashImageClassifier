import os
import warnings

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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


def apply_pca(data, n_components=2):
    """
    使用PCA进行降维
    """
    print("使用PCA进行降维")
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca, pca


def perform_clustering(data, num_clusters=2):
    """
    使用K均值聚类
    """
    print("使用K均值聚类")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(data)
    return predicted_labels, kmeans


def visualize_clusters(data, predicted_labels, num_clusters=2):
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


def evaluate_accuracy(predicted_labels, true_labels):
    """
    计算准确率
    """
    print("计算准确率")
    correct_predictions = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
    total_images = len(predicted_labels)
    accuracy = correct_predictions / total_images
    return accuracy


def map_labels_to_classes(true_labels, class_mapping):
    """
    映射聚类标签到真实标签
    """
    print("映射聚类标签到真实标签")
    mapped_labels = [class_mapping[1] if true == 1 else class_mapping[0] for true in true_labels]
    return mapped_labels


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
dataset_path = '../cc'

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
num_clusters = 2
predicted_labels, kmeans = perform_clustering(features_pca, num_clusters)

# 可视化聚类结果
visualize_clusters(features_pca, predicted_labels, num_clusters)

# 计算准确率
true_labels = [1 if i < 1000 else 0 for i in range(len(predicted_labels))]
accuracy = evaluate_accuracy(predicted_labels, true_labels)

print(f'准确率: {accuracy * 100:.2f}%')

# 预测标签到类别名称的映射
label_mapping = {0: '狗', 1: '猫'}

# 测试一个新图像
new_image_path = '../../test_images/cat_dog/dog.1003.jpg'
predicted_class = predict_image(new_image_path, model, pca, kmeans, label_mapping)

print(f'新图像的预测类别是：{predicted_class}')

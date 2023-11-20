# -*- coding: utf-8 -*-
"""
@File  : ui.py
@author: FxDr
@Time  : 2023/11/17 22:12
@Description:
"""
import sys
import warnings

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Microsoft YaHei'


def predict_image(image_path, model, pca, kmeans, label_mapping):
    # 读取和预处理图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    # 使用预训练的ResNet模型提取特征
    model.eval()
    with torch.no_grad():
        outputs = model(image)

    # 提取特征
    features = outputs.squeeze().numpy()
    # 使用PCA降维
    features_pca = pca.transform(features.reshape(1, -1))
    # 预测
    predicted_label = kmeans.predict(features_pca)[0]
    # 映射预测标签到类别名称
    predicted_class = label_mapping[predicted_label]

    return predicted_class


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


class ImagePredictor(QWidget):
    def __init__(self):
        super().__init__()

        self.result_label = None
        self.image_label = None
        self.predict_button = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('图像预测')
        self.setGeometry(100, 100, 600, 400)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        self.predict_button = QPushButton('选择图片进行预测', self)
        self.predict_button.clicked.connect(self.predict_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

    def predict_image(self):
        # options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog # 用原生的文件管理器
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像", r"X:\Coding\Github\PyTorch-ImageClassifier",
                                                   "图像文件 (*.png *.jpg *.bmp);;所有文件 (*)")

        if file_name:
            self.image_label.setText("预测中...")
            self.repaint()

            predicted_class = predict_image(file_name, model, pca, kmeans, label_mapping)

            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaledToWidth(300)
            self.image_label.setPixmap(pixmap)
            self.result_label.setText(f'预测结果：{predicted_class}')
            self.result_label.setStyleSheet('font-size: 16px; font-weight: bold; color: blue;')
            self.image_label.setAlignment(Qt.AlignCenter)
            self.result_label.setAlignment(Qt.AlignCenter)


if __name__ == '__main__':
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
    model.eval()

    # 提取特征
    features = []
    with torch.no_grad():
        for images in dataloader:
            outputs = model(images)
            features.extend(outputs.squeeze().numpy())

    features = torch.tensor(features)

    # 使用PCA降维
    pca = PCA(n_components=150)
    features_pca = pca.fit_transform(features)

    # 使用K均值聚类
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(features_pca)

    # 预测标签到类别名称的映射
    label_mapping = {0: '猫', 1: '狗'}

    app = QApplication(sys.argv)
    window = ImagePredictor()
    window.show()
    sys.exit(app.exec_())

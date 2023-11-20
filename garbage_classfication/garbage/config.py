# -*- coding: utf-8 -*-
"""
@File  : config.py
@author: FxDr
@Time  : 2023/11/12 21:29
@Description:
"""
import numpy as np
import torch
import torchvision.transforms as transforms  # transforms 模块包含用于图像预处理的各种转换操作
from PIL import Image
from torchvision.datasets import ImageFolder  # ImageFolder 类是用于加载具有类别标签的图像数据集的便捷工具
from torchvision.transforms import Lambda


def add_gaussian_noise(image, scale=5):
    image = np.array(image)
    h, w, c = image.shape
    noise = np.random.normal(0, scale, (h, w, c))
    noisy_image = np.clip(image + noise, 0, 255)
    return Image.fromarray(np.uint8(noisy_image))


def add_gaussian_noise_transform(x):
    return add_gaussian_noise(x, scale=5)


data_dir = r'X:\Coding\Github\PyTorch-ImageClassifier\garbage_classfication\Garbage classification\Garbage ' \
           r'classification'  # 数据目录
# 训练参数
num_epochs = 8  # 轮次
opt_func = torch.optim.Adam
# lr = 5.5e-5
lr = 6e-5

batch_size = 25  # 定义批量大小
split_sizes = [1590, 200, 737]  # 训练集验证集测试集图片数
transformers = {
    'demo1': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]),
    'demo2': transforms.Compose([
        transforms.Resize((256, 256)),
        # 随机图像亮度、对比度、饱和度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # 随机翻转
        # transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        # 随机放射变化
        transforms.RandomAffine(degrees=11, translate=(0.1, 0.1), scale=(0.8, 0.8)),
        transforms.ToTensor(),
        # 标准化图像，使用 ImageNet 的均值和标准差
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化
    ]),
    'demo3': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        Lambda(add_gaussian_noise_transform),  # 自定义噪声函数
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'demo4': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
}
# 大小调整为256x256像素，并且转换为Pytorch张量
# transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

datasets_dict = {
    '1': ImageFolder(data_dir, transform=transformers['demo1']),
    '2': ImageFolder(data_dir, transform=transformers['demo2']),
    '3': ImageFolder(data_dir, transform=transformers['demo3']),
    '4': ImageFolder(data_dir, transform=transformers['demo4']),
}

dataset = datasets_dict['1']
# dataset = datasets_dict['2']
# dataset = datasets_dict['3']
# dataset = datasets_dict['4']

# dataset1 = ImageFolder(data_dir, transform=transformers['demo1'])
# dataset2 = ImageFolder(data_dir, transform=transformers['demo2'])

# 创建示例输入张量
example_input = torch.randn(batch_size, 3, 256, 256)

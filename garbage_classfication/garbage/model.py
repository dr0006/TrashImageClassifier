# -*- coding: utf-8 -*-
"""
@File  : model.py
@author: FxDr
@Time  : 2023/11/12 21:21
@Description:
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet50_Weights

from config import dataset


def accuracy(outputs, labels):
    _, pred_s = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(torch.Tensor(pred_s == labels)).item() / len(pred_s), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, val_loader):
    """评估模型在验证集上的性能"""
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_default_device():
    """获取默认设备，如果有 GPU 则选择 GPU，否则选择 CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """将张量（或张量组成的列表/元组）移动到指定的设备"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """包装数据加载器，将数据移动到指定设备上"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """将数据移动到设备后，产生一个批次的数据"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """批次数量"""
        return len(self.dl)


class ImageClassificationBase(nn.Module):
    @staticmethod
    def print_dataset():
        print(dataset)

    @staticmethod
    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        """训练模型并在每个 epoch 结束后在验证集上评估性能"""
        history = []
        optimizer = opt_func(model.parameters(), lr, weight_decay=1e-5)
        start_time = time.time()  # 记录开始时间

        # 第一阶段：冻结层的初始训练
        print("第一阶段训练")
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # 每个 epoch 结束后的验证阶段
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)

            end_time = time.time()  # 记录每一轮的结束时间
            epoch_time = end_time - start_time  # 计算每一轮的时间
            print(f"第 {epoch + 1} 轮训练时间: {epoch_time // 60} 分 {epoch_time % 60} 秒")

            # 更新最佳模型参数
            if result['val_acc'] > model.best_accuracy:
                model.best_accuracy = result['val_acc']
                model.best_model_state = model.state_dict()

        model.load_state_dict(model.best_model_state)
        # 解冻层进行微调
        for param in model.parameters():
            param.requires_grad = True

        # 第二阶段：使用较低的学习率进行微调
        optimizer = opt_func(model.parameters(), lr=0.0001)
        print("第二阶段微调")
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # 每个 epoch 结束后的验证阶段
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)

            end_time = time.time()  # 记录每一轮的结束时间
            epoch_time = end_time - start_time  # 计算每一轮的时间
            print(f"第 {epoch + 1} 轮训练时间: {epoch_time // 60} 分 {epoch_time % 60} 秒")

            # 更新最佳模型参数
            if result['val_acc'] > model.best_accuracy:
                model.best_accuracy = result['val_acc']
                model.best_model_state = model.state_dict()

        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间
        print(f"训练总时间: {total_time // 60} 分 {total_time % 60} 秒")

        return history

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # 生成预测
        loss = f.cross_entropy(out, labels)  # 计算损失
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # 生成预测
        loss = f.cross_entropy(out, labels)  # 计算损失
        acc = accuracy(out, labels)  # 计算准确率
        return {'val_loss': loss.detach(), 'val_acc': acc}

    @staticmethod
    def validation_epoch_end(outputs):
        # 将损失列表转换为张量并计算均值
        batch_losses = torch.stack([x['val_loss'] for x in outputs])
        epoch_loss = batch_losses.mean()

        # 将准确率列表转换为张量并计算均值
        batch_acc_s = torch.stack([x['val_acc'] for x in outputs])
        epoch_acc = batch_acc_s.mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    @staticmethod
    def epoch_end(epoch, result):
        print(
            f"第 {epoch + 1} 轮: 训练损失: {result['train_loss']:.4f}, 验证损失: {result['val_loss']:.4f}, "
            f"验证准确率: {result['val_acc']:.4f}")


# 预训练模型 ResNet-50
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
        self.network.fc = nn.Linear(num_ftr_s, len(dataset.classes))

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


# 预训练GoogleNet
class GoogleNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.best_accuracy = 0.0
        self.best_model_state = None  # 最高测试集评分的参数副本

        self.network = models.googlenet(pretrained=True)

        # 替换最后一层全连接层
        num_ftr_s = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftr_s, len(dataset.classes))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

    def save_model(self):
        """保存模型"""
        # 转换百分比形式
        accuracy_percent = f'{self.best_accuracy * 100:.2f}%'
        # 保存模型的权重
        # torch.save(self.state_dict(), 'model/model_weights.pth')
        # 整个模型
        # torch.save(self, "model/entire_model.pth")
        # 保存最佳模型参数和整个模型
        best_model_weights_filename = f'model2/{accuracy_percent}_model_weights.pth'
        best_model_filename = f'model2/{accuracy_percent}_entire_model.pth'
        torch.save(self.best_model_state, best_model_weights_filename)
        torch.save(self, best_model_filename)

    def load_model_dict(self, path):
        """
        加载模型的权重
        :param path: 路径
        """
        self.load_state_dict(torch.load(path))
        self.eval()  # 设置为评估模式


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 当维度增加时的快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = f.dropout(out, p=0.5, training=self.training)  # 添加 dropout
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class CustomResNet(ImageClassificationBase):
    def __init__(self, pretrained=True):
        super(CustomResNet, self).__init__()
        self.best_accuracy = 0.0
        self.best_model_state = None

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块
        self.residual_block1 = ResidualBlock(64, 64)
        self.residual_block2 = ResidualBlock(64, 128, stride=2)
        self.residual_block3 = ResidualBlock(128, 256, stride=2)
        self.residual_block4 = ResidualBlock(256, 512, stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(512, len(dataset.classes))

        # 是否加载预训练权重
        if pretrained:
            resnet = models.resnet18(pretrained=True)
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.pool = resnet.maxpool

        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # 学习率调整
        self.scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return torch.sigmoid(x)

    def save_model(self):
        """保存模型"""
        # 转换百分比形式
        accuracy_percent = f'{self.best_accuracy * 100:.2f}%'
        # 保存最佳模型参数和整个模型
        best_model_weights_filename = f'../model/{accuracy_percent}_model_weights.pth'
        print(best_model_weights_filename)
        torch.save(self.best_model_state, best_model_weights_filename)

    def load_model_dict(self, path):
        """
        加载模型的权重
        :param path: 路径
        """
        self.load_state_dict(torch.load(path))
        self.eval()  # 设置为评估模式

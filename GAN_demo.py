# -*- coding: utf-8 -*-
"""
@File  : demo.py
@author: FxDr
@Time  : 2023/12/26 18:47
@Description:
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    # 设置超参数
    nz = 100  # 噪声向量的大小
    ngf = 64  # 生成器的特征图大小
    ndf = 64  # 判别器的特征图大小
    nc = 3  # 图像通道数

    # 初始化生成器和判别器
    generator = Generator(nz, ngf, nc)
    discriminator = Discriminator(nc, ndf)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 加载CIFAR-10数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # 训练模型
    num_epochs = 10

    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            real_images, _ = data
            batch_size = real_images.size(0)

            # 训练判别器
            discriminator.zero_grad()
            label_real = torch.full((batch_size,), 1.0)
            output_real = discriminator(real_images)
            loss_d_real = criterion(output_real.view(-1), label_real)
            loss_d_real.backward()

            noise = torch.randn(batch_size, nz, 1, 1)
            fake_images = generator(noise)
            label_fake = torch.full((batch_size,), 0.0)
            output_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(output_fake.view(-1), label_fake)
            loss_d_fake.backward()

            optimizer_d.step()

            # 训练生成器
            generator.zero_grad()
            output = discriminator(fake_images)
            loss_g = criterion(output.view(-1), label_real)
            loss_g.backward()
            optimizer_g.step()

            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         loss_d_real.item() + loss_d_fake.item(), loss_g.item()))

            # 保存生成图像
            if i % 500 == 0:
                torchvision.utils.save_image(fake_images[:6], f'./picgan/generated_images_epoch{epoch}_batch{i}.png',
                                             normalize=True, nrow=3)

    # 保存模型
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

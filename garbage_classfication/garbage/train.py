# -*- coding: utf-8 -*-
"""
@File  : train.py
@author: FxDr
@Time  : 2023/11/12 21:52
@Description:
"""

from torch.utils.data import DataLoader, random_split
# 自定义config和model
from garbage_classfication.garbage.config import dataset, batch_size, split_sizes, num_epochs, lr, opt_func
from garbage_classfication.garbage.model import ResNet, get_default_device, DeviceDataLoader, to_device
from garbage_classfication.garbage.tools.tools import plot_accuracies, plot_losses

if __name__ == '__main__':
    print('Training ResNet...')
    train_ds, val_ds, test_ds = random_split(dataset, split_sizes)
    print("训练集、验证集和测试集大小:")
    print(len(train_ds), len(val_ds), len(test_ds))

    # 创建 ResNet 模型实例
    model = ResNet()
    model.print_dataset()

    # 创建训练集 DataLoader
    # shuffle=True 表示在每个 epoch 开始时对数据进行洗牌 就是打乱的意思
    # num_workers 表示用于数据加载的子进程数量
    # pin_memory=True 可以将数据加载到 CUDA 固定内存中，提高数据传输速度
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=6, pin_memory=True)

    # 创建验证集 DataLoader
    # batch_size*2 验证集的批量大小是训练集的两倍
    val_dl = DataLoader(val_ds, batch_size * 2, num_workers=6, pin_memory=True)

    device = get_default_device()
    print("Using device:{}".format(device))
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)

    history = model.fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    plot_accuracies(history)
    plot_losses(history)

    save_model = input("是否保存模型？ (yes/no): ").lower()
    if save_model == 'yes':
        model.save_model()
        print("Model saved")
    else:
        print("Model is not saved")
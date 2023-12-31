# PyTorch-ImageClassifier

数字图像课设，图片分类  
选择垃圾分类

---

## 数据集选择

来自*Kaggle*的垃圾分类数据集：[下载地址](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/code)  
一共有六类垃圾:纸板（cardboard）、玻璃（glass）、金属（metal）、纸（paper）、塑料（plastic）、其他垃圾（trash）  
393（cardboard） + 491（glass） + 400（metal） + 584（paper） + 472（plastic） + 127（trash） = 2467 张  
403 501 410 594 482 137 = 2527

Garbage Classification Data  
The Garbage Classification Dataset contains 6 classifications:

1. cardboard (393)
2. glass (491)
3. metal (400)
4. paper(584)
5. plastic (472)
6. trash(127).

### 垃圾分类（12类）

另一个数据集：[下载地址](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

ImageFolder类还会自动为每个图像分配一个标签。标签是基于图像文件夹的子文件夹的名称分配的。

```

/Garbage classification/

cardboard/
cardboard1.jpg
cardboard2.jpg

glass/
glass1.jpg
glass2.jpg
```

则ImageFolder将为cardboard文件夹中的所有图像分配标签0，将为glass文件夹中的所有图像分配标签1

```
import torchvision.transforms as transforms  # transforms 模块包含用于图像预处理的各种转换操作
from torchvision.datasets import ImageFolder  # ImageFolder 类是用于加载具有类别标签的图像数据集的便捷工具

ImageFolder(data_dir, transform=transformers['demo1']),
```

```markdown
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
}
```

### 猫狗

数据集：[下载地址](https://www.kaggle.com/datasets/lizhensheng/-2000/discussion)

## 问题

就是用class_index 前面没发现问题，是因为刚好和正确的顺序一样

```markdown
# ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

class_index = ['纸板', '玻璃', '金属', '纸', '塑料', '其他垃圾']

# ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']

class_index2 = ['金属', '玻璃', '纸', '其他垃圾', '纸板', '塑料']
```

```python
def predict_img(path, model):
    device = get_default_device()
    image = Image.open(path)
    # 大小调整为256x256像素，并且转换为Pytorch张量
    image_clean = transformers['demo4'](image)
    # 将图像转换为张量并移动到device
    xb = to_device(image_clean.unsqueeze(0), device)
    # 预测
    yb = model(xb)
    # 选择概率最高的类别
    prob, preds = torch.max(yb, dim=1)
    # print(prob.item())
    index = preds[0].item()
    print("index", index)
    label = dataset.classes[index]
    print("label", label)
    print(dataset.classes)
    # show_pred_image(image, index, label)
    return label, prob.item()
```


class CNN_V1(nn.Module):
"""
添加一个隐藏层、调整 dropout 值、增加一个卷积层
总共 3 个隐藏层、3 个卷积层和批量归一化
"""

    # 构造函数
    def __init__(self, out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0):
        super(CNN_V1, self).__init__()

        # 第一个卷积层
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.drop_conv = nn.Dropout(p=0.2)

        # 第二个卷积层
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # 第三个卷积层
        self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2)
        self.conv3_bn = nn.BatchNorm2d(out_3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # 隐藏层 1
        self.fc1 = nn.Linear(out_3 * 4 * 4, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        self.drop = nn.Dropout(p=p)

        # 隐藏层 2
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)

        # 隐藏层 3
        self.fc3 = nn.Linear(1000, 1000)
        self.fc3_bn = nn.BatchNorm1d(1000)

        # 隐藏层 4
        self.fc4 = nn.Linear(1000, 1000)
        self.fc4_bn = nn.BatchNorm1d(1000)

        # 最终输出层
        self.fc5 = nn.Linear(1000, number_of_classes)
        self.fc5_bn = nn.BatchNorm1d(number_of_classes)

    # 前向传播
    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.drop_conv(x)

        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.drop_conv(x)

        x = self.cnn3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.drop_conv(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(self.drop(x))

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(self.drop(x))

        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(self.drop(x))

        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = F.relu(self.drop(x))

        x = self.fc5(x)
        x = self.fc5_bn(x)

        return x

# model = CNN_V1(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5)



```markdown
 # ---------------------  Model Net -------------------------------
    # 构建模型
    print('==> Building model..')
    net = ResNet18()
    # net = MyResNet(3, 10)
    # net = CNN_V1(out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0.5)
    # ----------------------------------------------------------------

    # 使用DataParallel在多GPU上运行
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = DataParallel(net)

    net = net.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # 定义学习率调整策略
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # TensorBoard可视化
    # writer = SummaryWriter()
    writer = SummaryWriter(comment="6")

    # 定义无标签数据集和加载器
    unlabeled_dataset = torchvision.datasets.CIFAR10(
        root='../../data', train=True, download=False, transform=transform_train)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=128, shuffle=True, num_workers=4)

    # 训练循环
    for epoch in range(start_epoch, start_epoch + 200):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        # 混合有标签和无标签数据
        mixed_loader = zip(train_loader, unlabeled_loader)

        for (inputs, targets), (unlabeled_inputs, _) in mixed_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            unlabeled_inputs = unlabeled_inputs.to(device)

            optimizer.zero_grad()

            # 计算有标签数据的损失
            outputs = net(inputs)
            loss_supervised = criterion(outputs, targets)

            # 生成伪标签并计算无标签数据的损失
            pseudo_labels = generate_pseudo_labels(net, unlabeled_loader, device)
            pseudo_dataset = torch.utils.data.TensorDataset(unlabeled_dataset.data, pseudo_labels)
            pseudo_loader = torch.utils.data.DataLoader(
                pseudo_dataset, batch_size=128, shuffle=True, num_workers=4)

            for pseudo_inputs, pseudo_targets in pseudo_loader:
                pseudo_inputs, pseudo_targets = pseudo_inputs.to(device), pseudo_targets.to(device)
                pseudo_outputs = net(pseudo_inputs)
                loss_unsupervised = criterion(pseudo_outputs, pseudo_targets)

            # 总损失为有标签数据损失和无标签数据损失的加权和
            alpha = 0.1  # 超参数，控制有标签数据损失和无标签数据损失的权重
            loss = (1 - alpha) * loss_supervised + alpha * loss_unsupervised

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_accuracy = 100. * correct / total

        # 验证模型
        net.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validation_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_accuracy = 100. * correct / total

```
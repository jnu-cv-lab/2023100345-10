# -*- coding: utf-8 -*-
"""
第10次实验：CNN训练过程分析、优化器对比、卷积核/特征图可视化、错误样本分析
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 1. 定义CNN模型（与上次实验相同）==========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # 计算全连接层输入维度
        self._to_linear = None
        self._compute_linear_size()
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
       

    def _compute_linear_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========== 数据加载 ==========
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ========== 训练函数（记录loss和accuracy）==========
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, optimizer_name='Adam', momentum=0.9):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD+Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, epochs+1):
        # 训练
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"{optimizer_name} lr={lr} Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}, Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")
    
    # 测试
    test_acc = test_model(model, test_loader)
    return train_losses, val_losses, train_accs, val_accs, test_acc

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total

# ========== 任务2：优化器对比 ==========
def compare_optimizers():
    print("\n" + "="*50)
    print("任务2：优化器对比（SGD, SGD+Momentum, Adam）")
    train_loader, val_loader, test_loader = get_mnist_loaders()
    optimizers = ['SGD', 'SGD+Momentum', 'Adam']
    results = {}
    for opt in optimizers:
        print(f"\n--- 训练 {opt} ---")
        model = SimpleCNN()
        train_losses, val_losses, train_accs, val_accs, test_acc = train_model(
            model, train_loader, val_loader, epochs=5, lr=0.01 if opt=='SGD' else 0.001, optimizer_name=opt
        )
        results[opt] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_acc': test_acc
        }
        print(f"{opt} 测试准确率: {test_acc:.4f}")
    # 绘制对比曲线
    plt.figure(figsize=(12,4))
    for opt in optimizers:
        plt.subplot(1,2,1)
        plt.plot(range(1,6), results[opt]['val_losses'], label=opt)
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(range(1,6), results[opt]['val_accs'], label=opt)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    plt.show()
    return results

# ========== 任务3：学习率对比（Adam）==========
def compare_lr():
    print("\n" + "="*50)
    print("任务3：学习率对比（Adam, lr=0.1, 0.01, 0.001）")
    train_loader, val_loader, test_loader = get_mnist_loaders()
    lrs = [0.1, 0.01, 0.001]
    results = {}
    for lr in lrs:
        print(f"\n--- 训练 Adam lr={lr} ---")
        model = SimpleCNN()
        train_losses, val_losses, train_accs, val_accs, test_acc = train_model(
            model, train_loader, val_loader, epochs=5, lr=lr, optimizer_name='Adam'
        )
        results[lr] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_acc': test_acc
        }
        print(f"Adam lr={lr} 测试准确率: {test_acc:.4f}")
    # 绘制对比曲线
    plt.figure(figsize=(12,4))
    for lr in lrs:
        plt.subplot(1,2,1)
        plt.plot(range(1,6), results[lr]['val_losses'], label=f'lr={lr}')
        plt.title('Validation Loss (Adam)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(range(1,6), results[lr]['val_accs'], label=f'lr={lr}')
        plt.title('Validation Accuracy (Adam)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.tight_layout()
    plt.savefig('lr_comparison.png')
    plt.show()
    return results

# ========== 任务4：卷积核可视化 ==========
def visualize_filters(model, save_path='conv1_filters.png'):
    print("\n" + "="*50)
    print("任务4：第一层卷积核可视化")
    # 获取第一层卷积核权重 shape: [32, 1, 3, 3]
    filters = model.conv1.weight.data.cpu().numpy()
    # 显示前16个（至少8个）
    num_filters = min(16, filters.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(8,8))
    for i in range(num_filters):
        ax = axes[i//4, i%4]
        filt = filters[i, 0]  # 单通道
        ax.imshow(filt, cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    plt.suptitle('Conv1 Filters')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"卷积核图已保存为 {save_path}")
    print("分析：训练后的卷积核呈现出边缘检测、方向或纹理特征，因为网络通过反向传播自动学习到了这些模式。")

# ========== 任务5：Feature map 可视化 ==========
def visualize_feature_maps(model, test_loader, save_path='feature_maps.png'):
    print("\n" + "="*50)
    print("任务5：第一层卷积输出的特征图可视化")
    model.eval()
    # 获取一张测试图片
    images, labels = next(iter(test_loader))
    img = images[0:1].to(device)  # 取一张图
    # 注册 hook 获取第一层卷积的输出
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.conv1.register_forward_hook(get_activation('conv1'))
    _ = model(img)
    conv1_out = activation['conv1'].squeeze()  # shape: [32, 28, 28] (因为padding=1且步长1，输出大小不变)
    # 显示前16个 feature maps
    num_maps = min(16, conv1_out.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(10,10))
    for i in range(num_maps):
        ax = axes[i//4, i%4]
        fm = conv1_out[i].cpu().numpy()
        ax.imshow(fm, cmap='viridis')
        ax.set_title(f'Map {i+1}')
        ax.axis('off')
    plt.suptitle('Feature Maps after Conv1')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"特征图已保存为 {save_path}")
    print("观察：不同特征图响应不同区域，有的对边缘敏感，有的对纹理敏感，说明卷积核提取了多样化的特征。")

# ========== 任务6：错误分类样本分析 ==========
def show_error_samples(model, test_loader, num_samples=8, save_path='error_samples.png'):
    print("\n" + "="*50)
    print("任务6：错误分类样本分析")
    model.eval()
    error_images = []
    error_trues = []
    error_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    error_images.append(images[i].cpu())
                    error_trues.append(labels[i].item())
                    error_preds.append(preds[i].item())
                    if len(error_images) >= num_samples:
                        break
            if len(error_images) >= num_samples:
                break
    # 显示错误样本
    fig, axes = plt.subplots(2, num_samples//2, figsize=(12, 5))
    axes = axes.flatten()
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(error_images[i].squeeze(), cmap='gray')
        ax.set_title(f"True:{error_trues[i]}/Pred:{error_preds[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"错误样本图已保存为 {save_path}")
    # 统计混淆对
    from collections import Counter
    confusions = Counter(zip(error_trues, error_preds))
    print("最易混淆的类别对（真实->预测）：")
    for (true, pred), cnt in confusions.most_common(5):
        print(f"  {true} -> {pred} : {cnt} 次")
    print("错误原因分析：数字形状相似（如4和9、7和1、3和8），或者书写风格导致特征不明显。改进建议：数据增强（旋转、缩放）、增加网络深度、使用Dropout等正则化。")

# ========== 任务7：混淆矩阵 ==========
def plot_confusion_matrix(model, test_loader, save_path='confusion_matrix.png'):
    print("\n" + "="*50)
    print("任务7：混淆矩阵")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()
    print(f"混淆矩阵已保存为 {save_path}")
    print("对角线元素代表正确分类的样本数，非对角线元素代表错误分类的样本数。")
    # 找出非对角线最大值（排除对角线）
    np.fill_diagonal(cm, 0)
    max_idx = np.unravel_index(np.argmax(cm), cm.shape)
    print(f"混淆最严重的类别对：{max_idx[0]} 和 {max_idx[1]}，错误次数 {cm[max_idx]}")
    return cm

# ========== 主函数 ==========
if __name__ == "__main__":
    # 先训练一个模型用于可视化和错误分析（使用Adam默认参数）
    print("先训练一个基准模型（Adam, lr=0.001, 5 epochs）用于后续可视化...")
    train_loader, val_loader, test_loader = get_mnist_loaders()
    model = SimpleCNN()
    # 简单训练5轮
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 6):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} completed")
    test_acc = test_model(model, test_loader)
    print(f"基准模型测试准确率: {test_acc:.4f}")
    
    # 任务4：卷积核可视化
    visualize_filters(model)
    
    # 任务5：特征图可视化
    visualize_feature_maps(model, test_loader)
    
    # 任务6：错误样本分析
    show_error_samples(model, test_loader)
    
    # 任务7：混淆矩阵
    plot_confusion_matrix(model, test_loader)
    
    # 任务2和任务3需要单独运行，因为会重新训练多个模型，耗时较长
    # 取消下面的注释即可运行（可以在需要时运行）
    # compare_optimizers()
    # compare_lr()
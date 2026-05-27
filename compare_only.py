# compare_only.py - 只做优化器对比和学习率对比
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义模型（与原来相同）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self._to_linear = None
        self._compute_linear()
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    def _compute_linear(self):
        with torch.no_grad():
            x = torch.zeros(1,1,28,28)
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

def get_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_size = int(0.8 * len(full))
    val_size = len(full) - train_size
    train_set, val_set = random_split(full, [train_size, val_size])
    return DataLoader(train_set, batch_size, shuffle=True), DataLoader(val_set, batch_size), DataLoader(test, batch_size)

def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, optimizer_name='Adam'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD+Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(1, epochs+1):
        model.train()
        running_loss, correct, total = 0,0,0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(images)
            _, pred = out.max(1)
            correct += (pred == labels).sum().item()
            total += len(images)
        train_loss = running_loss / total
        train_acc = correct / total
        model.eval()
        val_loss, val_correct, val_total = 0,0,0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = criterion(out, labels)
                val_loss += loss.item() * len(images)
                _, pred = out.max(1)
                val_correct += (pred == labels).sum().item()
                val_total += len(images)
        val_loss /= val_total
        val_acc = val_correct / val_total
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        print(f"{optimizer_name} lr={lr} Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    # test
    test_loader = get_loaders()[2]
    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            _, pred = out.max(1)
            correct += (pred == labels).sum().item()
            total += len(images)
    test_acc = correct / total
    print(f"{optimizer_name} lr={lr} Test Acc: {test_acc:.4f}")
    return train_losses, val_losses, train_accs, val_accs, test_acc

# 任务2：优化器对比
print("="*50)
print("任务2：优化器对比")
train_loader, val_loader, _ = get_loaders()
optimizers = [('SGD', 0.01), ('SGD+Momentum', 0.01), ('Adam', 0.001)]
opt_results = {}
for name, lr in optimizers:
    print(f"\n--- 训练 {name} lr={lr} ---")
    model = SimpleCNN()
    train_losses, val_losses, train_accs, val_accs, test_acc = train_model(model, train_loader, val_loader, epochs=5, lr=lr, optimizer_name=name)
    opt_results[name] = {'train_losses': train_losses, 'val_losses': val_losses, 'train_accs': train_accs, 'val_accs': val_accs, 'test_acc': test_acc}
# 绘图
plt.figure(figsize=(12,4))
for name in opt_results:
    plt.subplot(1,2,1); plt.plot(range(1,6), opt_results[name]['val_losses'], label=name)
    plt.subplot(1,2,2); plt.plot(range(1,6), opt_results[name]['val_accs'], label=name)
plt.subplot(1,2,1); plt.title('Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.subplot(1,2,2); plt.title('Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.tight_layout(); plt.savefig('optimizer_comparison.png'); plt.show()

# 任务3：学习率对比
print("\n"+"="*50)
print("任务3：学习率对比 (Adam)")
lrs = [0.1, 0.01, 0.001]
lr_results = {}
for lr in lrs:
    print(f"\n--- 训练 Adam lr={lr} ---")
    model = SimpleCNN()
    train_losses, val_losses, train_accs, val_accs, test_acc = train_model(model, train_loader, val_loader, epochs=5, lr=lr, optimizer_name='Adam')
    lr_results[lr] = {'train_losses': train_losses, 'val_losses': val_losses, 'train_accs': train_accs, 'val_accs': val_accs, 'test_acc': test_acc}
plt.figure(figsize=(12,4))
for lr in lrs:
    plt.subplot(1,2,1); plt.plot(range(1,6), lr_results[lr]['val_losses'], label=f'lr={lr}')
    plt.subplot(1,2,2); plt.plot(range(1,6), lr_results[lr]['val_accs'], label=f'lr={lr}')
plt.subplot(1,2,1); plt.title('Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.subplot(1,2,2); plt.title('Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.tight_layout(); plt.savefig('lr_comparison.png'); plt.show()

print("优化器对比和学习率对比完成！")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # 推荐用于显示训练进度
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random # 导入 random 库
import torchvision.models as models
from torch.optim import lr_scheduler # 导入学习率调度器模块

# --- 配置设备 ---
# 使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 此外还可以使用os.environ来控制GPU可见性，但通常在脚本开始前设置：
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 如果要用0号GPU

# --- 配置其他超参数 ---
batch_size = 64
num_workers = 0 # num_workers: 根据操作系统和并行需求配置
                # 对于 Windows 系统，通常必须设置为 0
                # 对于 Linux/macOS，如果希望并行加载数据，可以设置为 CPU 核心数的一半或更少
lr = 1e-4       # 学习率
epochs = 20     # 训练轮数

# --- 早停参数 ---
patience = 7    # 在验证集性能连续多少个epoch没有提升时停止
min_delta = 0.0001 # 性能提升的最小阈值，小于这个值不算提升

# --- 学习率调度器参数 ---
lr_factor = 0.5  # 学习率每次降低的因子
lr_patience = 3  # 在验证集性能连续多少个epoch没有提升时降低学习率

# --- 设置随机种子，确保结果可以复现 ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # 以下两行可以帮助获得更可复现的结果，但可能会降低一点点性能
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(42) # 在脚本的开头调用随机种子函数

# --- 数据预处理和增强 ---
image_size = 28 # FashionMNIST 图像尺寸

data_transform = transforms.Compose([
    transforms.ToPILImage(),         # 将 numpy 数组转换为 PIL 图像
    transforms.Resize(image_size),   # 调整图像大小
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
    transforms.RandomRotation(10),   # 随机旋转
    transforms.RandomCrop(image_size, padding=4), # 随机裁剪
    transforms.ToTensor(),           # 将 PIL 图像或 numpy.ndarray 转换为 Tensor，并归一化到 [0.0, 1.0]
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 将单通道图像复制为三通道，以适应预训练模型输入
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)) # MNIST/FashionMNIST标准化参数，针对三通道
])

# --- 自定义数据集类 ---
class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        # 将像素值转换为 numpy.uint8 类型
        self.images = df.iloc[:, 1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 图像数据是扁平的，需要重塑为 (H, W, C) -> (28, 28, 1)
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)
        else:
            # 如果没有提供 transform，则手动转换为 Tensor 并归一化
            # 注意：如果使用了 transform，此分支不会被执行
            image = torch.tensor(image / 255., dtype=torch.float).permute(2, 0, 1) # 转换为 (C, H, W)
            image = image.repeat(3, 1, 1) # 如果没有transform，也需要转换为3通道
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# --- 加载数据集 ---
# 确保你的 CSV 文件路径正确
train_df = pd.read_csv("/home/jovyan/work/datasets/6645b9bbc4db0a94029f0bfe-momodel/FashionMNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("/home/jovyan/work/datasets/6645b9bbc4db0a94029f0bfe-momodel/FashionMNIST/fashion-mnist_test.csv")

train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# --- 定义模型 ---
# 这里我们选择使用 ResNet18 作为示例模型，并对其进行适配
# 你可以根据需要替换成你想要的任何其他 torchvision 预训练模型
# 例如：model = models.alexnet(pretrained=True)
# 注意：不同的模型其分类层的名称和结构可能不同，需要查看其源码或使用 print(model) 来确定
model = models.resnet18(pretrained=True)

# 获取ResNet18全连接层（分类层）的输入特征数量
num_ftrs = model.fc.in_features

# 将原始分类层替换为适用于FashionMNIST 10个类别的新的全连接层
model.fc = nn.Linear(num_ftrs, 10) # FashionMNIST有10个类别

# 将模型移动到指定的设备 (CPU 或 GPU)
model = model.to(device)
print(model)

# --- 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss()
# 如果需要类别不平衡加权，可以像你原来那样设置：
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1,3,1,1,1,1,1], dtype=torch.float).to(device))

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) # 使用定义的 lr 变量

# --- 初始化学习率调度器 ---
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=lr_factor, patience=lr_patience, verbose=True
)

# --- 训练函数 ---
def train(epoch):
    model.train() # 设置模型为训练模式
    total_train_loss = 0 # 将 train_loss 重命名为 total_train_loss，避免与函数返回值混淆
    # 使用 tqdm 显示进度条
    with tqdm(train_loader, desc=f"Epoch {epoch} Training") as pbar:
        for data, label in pbar:
            data, label = data.to(device), label.to(device) # 将数据和标签移动到设备

            optimizer.zero_grad() # 梯度清零
            output = model(data)  # 前向传播
            loss = criterion(output, label) # 计算损失
            loss.backward()       # 反向传播
            optimizer.step()      # 更新模型参数

            total_train_loss += loss.item() * data.size(0) # 累加损失

            pbar.set_postfix({'Loss': loss.item()}) # 在进度条上显示当前批次的损失

    average_train_loss = total_train_loss / len(train_loader.dataset) # 计算平均训练损失
    print(f'Epoch: {epoch} \tTraining Loss: {average_train_loss:.6f}')
    return average_train_loss # 返回平均训练损失
    
# --- 验证函数 ---
def val(epoch):
    model.eval() # 设置模型为评估模式
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad(): # 在验证阶段不计算梯度
        with tqdm(test_loader, desc=f"Epoch {epoch} Validation") as pbar:
            for data, label in pbar:
                data, label = data.to(device), label.to(device) # 将数据和标签移动到设备
                
                output = model(data)        # 前向传播
                preds = torch.argmax(output, 1) # 获取预测结果（概率最大的类别）

                gt_labels.append(label.cpu().data.numpy()) # 收集真实标签
                pred_labels.append(preds.cpu().data.numpy()) # 收集预测标签

                loss = criterion(output, label) # 计算损失
                val_loss += loss.item() * data.size(0) # 累加损失

                pbar.set_postfix({'Loss': loss.item()}) # 在进度条上显示当前批次的损失

    val_loss = val_loss / len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    print(f'Epoch: {epoch} \tValidation Loss: {val_loss:.6f}, Accuracy: {acc:.6f}')
    return val_loss, acc # 返回验证损失和准确率

# --- 训练循环 ---
best_val_loss = float('inf') # 初始化最佳验证损失为无穷大
epochs_no_improve = 0        # 记录验证性能没有提升的 epoch 数量
early_stop_flag = False      # 早停标志

history = {'train_loss': [], 'val_loss': [], 'val_acc': []} # 存储训练历史

print("\n--- Starting Training ---")
for epoch in range(1, epochs + 1):
    # 捕获 train 函数的返回值
    train_loss = train(epoch) 
    val_loss, val_acc = val(epoch)  # val函数现在返回损失和准确率

    history['train_loss'].append(train_loss) # 现在 train_loss 已经被定义并赋值
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # 更新学习率
    scheduler.step(val_loss)
    
    # 打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} | Current LR: {current_lr:.6f}")

    # --- 早停逻辑 ---
    if val_loss < best_val_loss - min_delta: # 如果当前验证损失显著优于历史最佳
        best_val_loss = val_loss
        epochs_no_improve = 0 # 重置没有提升的 epoch 计数
        # 可选：保存当前最佳模型
        # 确保保存路径的目录存在，否则会报错
        os.makedirs('/home/jovyan/work/results', exist_ok=True) 
        torch.save(model.state_dict(), '/home/jovyan/work/results/best_fashion_model.pth')
        print("Validation loss improved. Saving best model state_dict.")
    else:
        epochs_no_improve += 1 # 否则，没有提升的 epoch 计数加一
        print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
        if epochs_no_improve >= patience: # 如果连续没有提升的 epoch 数量达到 patience
            print(f"Early stopping at epoch {epoch} as validation loss did not improve for {patience} consecutive epochs.")
            early_stop_flag = True
            break # 退出训练循环

    if early_stop_flag:
        break

print("\n--- Training Finished ---")

# --- 保存最终模型 ---
# 可以选择保存最佳模型（通过state_dict）或者训练结束时的模型
# 如果你在早停逻辑中保存了 best_model.pth，这里可以跳过
save_path = "/home/jovyan/work/results/FahionModel_final.pkl"
os.makedirs(os.path.dirname(save_path), exist_ok=True) # 确保保存路径的目录存在
torch.save(model.state_dict(), save_path) # 通常建议保存 state_dict 而不是整个模型
print(f"Final model state_dict saved to {save_path}")

# --- 绘制训练历史曲线 ---
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Training Loss') # 添加训练损失曲线
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('/home/jovyan/work/results/training_history.png')
plt.show()
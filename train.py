import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from config import Config
from data_loader import get_loaders
from models.MDR_UNet import get_model
from losses import CombinedLoss
from utils import calculate_metrics


# 绘图函数

def plot_curves(train_losses, val_dices, val_ious, epoch):
    """
    绘制训练Loss和验证指标曲线

    """
    plt.figure(figsize=(12, 5))

    # 左边：Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'r-', label='Train Loss')
    plt.title(f'Training Loss (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 右边：Dice & IoU
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_dices) + 1), val_dices, 'b-', label='Val Dice')
    plt.plot(range(1, len(val_ious) + 1), val_ious, 'g--', label='Val IoU')
    plt.title(f'Validation Metrics (Epoch {epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(Config.PLOT_DIR, f"{Config.MODEL_NAME}_training_curves.png")
    plt.savefig(save_path)
    plt.close()


# 核心训练逻辑

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, leave=True)
    epoch_loss = 0

    for data, targets in loop:
        data = data.to(Config.DEVICE)
        targets = targets.float().unsqueeze(1).to(Config.DEVICE)

        # 混合精度训练
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)


def validate(loader, model):
    model.eval()

    dice_score = 0.0
    iou_score = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(Config.DEVICE)
            targets = targets.float().unsqueeze(1).to(Config.DEVICE)

            preds = model(data)

            dice, iou, _, _ = calculate_metrics(preds, targets)

            dice_score += dice
            iou_score += iou
            num_batches += 1

    model.train()

    return dice_score / num_batches, iou_score / num_batches


def main():
    print(f"开始训练: {Config.MODEL_NAME}")

    # 1. 准备
    model = get_model().to(Config.DEVICE)
    #损失函数的更改位置，
    loss_fn = CombinedLoss(loss_name="bce_tversky")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    train_loader, val_loader = get_loaders()

    # 2.记录器
    best_dice = 0
    train_losses = []
    val_dices = []
    val_ious = []

    train_start_time = time.perf_counter()

    # 3.循环
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{Config.NUM_EPOCHS}]")

        # 训练
        loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)

        # 验证
        dice, iou = validate(val_loader, model)

        # 记录
        train_losses.append(loss)
        val_dices.append(dice)
        val_ious.append(iou)

        print(f"   >>> Val Dice: {dice:.4f} | IoU: {iou:.4f}")

        # 绘图
        plot_curves(train_losses, val_dices, val_ious, epoch)

        # 保存最佳
        if dice > best_dice:
            best_dice = dice
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.MODEL_NAME}_best.pth")
            torch.save(model.state_dict(), save_path)
            print("最佳模型已保存!")
    # 4. 统计总训练时间
    total_train_time_min = (time.perf_counter() - train_start_time) / 60

    runtime_path = os.path.join(Config.RESULTS_DIR, f"{Config.MODEL_NAME}_runtime.txt")
    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write(f"{total_train_time_min:.4f}")

    print("=" * 50)
    print(f"总训练时间 Run Time: {total_train_time_min:.4f} min")
    print(f"训练时间已保存至: {runtime_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
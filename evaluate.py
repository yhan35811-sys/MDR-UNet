import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

from config import Config
from data_loader import get_loaders
from models.MDR_UNet import get_model
from utils import calculate_metrics


def unnormalize(img_tensor):
    """反归一化"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def evaluate():
    print(f"正在评估模型: {Config.MODEL_NAME}")

    # 1. 加载模型
    model = get_model().to(Config.DEVICE)
    weight_path = f"{Config.CHECKPOINT_DIR}/{Config.MODEL_NAME}_best.pth"

    if not os.path.exists(weight_path):
        print(f"错误：找不到权重文件 {weight_path}")
        return

    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # 2. 准备数据
    _, val_loader = get_loaders()

    # 第一步：计算全集平均指标
    print(f"正在计算验证集整体指标 (共 {len(val_loader.dataset)} 张图片)...")

    avg_dice = 0
    avg_iou = 0
    avg_prec = 0
    avg_rec = 0


    steps = 0

    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc="Evaluating"):
            data = data.to(Config.DEVICE)
            targets = targets.float().unsqueeze(1).to(Config.DEVICE)

            preds = model(data)

            # 计算批次指标
            dice, iou, prec, rec = calculate_metrics(preds, targets)


            avg_dice += dice
            avg_iou += iou
            avg_prec += prec
            avg_rec += rec

            steps += 1

    # 计算平均值
    final_dice = avg_dice / steps
    final_iou = avg_iou / steps
    final_prec = avg_prec / steps
    final_rec = avg_rec / steps


    # 直接打印最终平均结果
    print("\n" + "=" * 50)
    print(f" 最终测试结果 (Model: {Config.MODEL_NAME})")
    print("=" * 50)
    print(f"Dice Score : {final_dice:.4f}")
    print(f"IoU Score  : {final_iou:.4f}")
    print(f"Precision  : {final_prec:.4f}")
    print(f"Recall     : {final_rec:.4f}")

    print("=" * 50 + "\n")

    # 第二步：生成固定样本对比图
    indices = Config.VIS_INDICES
    print(f"正在生成固定样本对比图 (Indices: {indices})...")

    val_ds = val_loader.dataset
    plt.figure(figsize=(10, 15))  # 5行3列

    for i, idx in enumerate(indices):
        if idx >= len(val_ds): idx = 0

        # 获取单张数据
        img_tensor, mask_numpy = val_ds[idx]
        input_tensor = img_tensor.unsqueeze(0).to(Config.DEVICE)

        # 预测
        with torch.no_grad():
            pred_logits = model(input_tensor)
            pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()

        # 绘图准备
        img_vis = unnormalize(img_tensor)
        mask_vis = mask_numpy
        pred_vis = pred_mask[0][0].cpu().numpy()

        # 1. 原图
        plt.subplot(len(indices), 3, i * 3 + 1)
        plt.imshow(img_vis)
        plt.axis('off')
        if i == 0: plt.title("Original Image", fontsize=12, fontweight='bold')
        plt.text(-30, 128, f"Case {idx}", fontsize=12, rotation=90, verticalalignment='center')

        # 2. 标签
        plt.subplot(len(indices), 3, i * 3 + 2)
        plt.imshow(mask_vis, cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Ground Truth", fontsize=12, fontweight='bold')

        # 3. 预测
        plt.subplot(len(indices), 3, i * 3 + 3)
        plt.imshow(pred_vis, cmap='gray')
        plt.axis('off')
        if i == 0: plt.title(f"Prediction\n({Config.MODEL_NAME})", fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = f"{Config.VIS_RESULT_DIR}/Result_{Config.MODEL_NAME}_Fixed5.png"
    plt.savefig(save_path, dpi=300)
    print(f"可视化图片已保存至: {save_path}")


if __name__ == "__main__":
    evaluate()

import numpy as np
import torch

def calculate_metrics(pred, target, threshold=0.5):
    # Pred: Logits -> Sigmoid -> Binary
    pred = (torch.sigmoid(pred) > threshold).float()

    pred = pred.view(-1)
    target = target.view(-1)

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    # Dice & IoU
    smooth = 1e-5
    dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)

    # Precision & Recall
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)

    return dice.item(), iou.item(), precision.item(), recall.item()




def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)
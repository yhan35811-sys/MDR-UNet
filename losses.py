import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Dice Loss
# =========================================================

class DiceLoss(nn.Module):
    """
    Dice Loss 主要看预测区域和真实区域的重叠程度。
    适合医学图像分割，尤其适合前景区域很小的情况。
    """

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred 是 logits，所以这里先 sigmoid
        pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()

        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1.0 - dice


# =========================================================
# 2. Binary Cross Entropy Loss
# =========================================================

class BCELoss(nn.Module):
    """
    BCE Loss 是最基础的二分类损失。
    注意这里用 BCEWithLogitsLoss，所以模型最后不要加 sigmoid。
    """

    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


# =========================================================
# 3. Weighted BCE Loss
# =========================================================

class WeightedBCELoss(nn.Module):
    """
    Weighted BCE 用来处理前景和背景不平衡。

    pos_weight 越大，模型越重视前景伤口区域。
    如果你的模型漏分伤口，可以适当增大 pos_weight。
    """

    def __init__(self, pos_weight=3.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight_value = pos_weight

    def forward(self, pred, target):
        pos_weight = torch.tensor(
            [self.pos_weight_value],
            device=pred.device
        )

        loss = F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=pos_weight
        )

        return loss


# =========================================================
# 4. Tversky Loss
# =========================================================

class TverskyLoss(nn.Module):
    """
    Tversky Loss 是 Dice Loss 的改进版。

    alpha 控制 FP，也就是误分背景为伤口。
    beta 控制 FN，也就是漏掉真实伤口。

    如果你的模型漏分严重，应该让 beta 更大。
    对 DFU 分割，一般推荐：
        alpha = 0.3
        beta = 0.7
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1)

        TP = (pred * target).sum()
        FP = (pred * (1 - target)).sum()
        FN = ((1 - pred) * target).sum()

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1.0 - tversky


# =========================================================
# 5. Combined Loss，可以自由切换
# =========================================================

class CombinedLoss(nn.Module):
    """
    这里通过 loss_name 控制使用哪种损失函数。

    可选：
        "dice"
        "bce"
        "weighted_bce"
        "tversky"
        "bce_dice"
        "weighted_bce_dice"
        "bce_tversky"
        "dice_tversky"
    """

    def __init__(self, loss_name="bce_dice"):
        super(CombinedLoss, self).__init__()

        self.loss_name = loss_name

        self.dice = DiceLoss()
        self.bce = BCELoss()
        self.weighted_bce = WeightedBCELoss(pos_weight=3.0)
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, pred, target):

        if self.loss_name == "dice":
            return self.dice(pred, target)

        elif self.loss_name == "bce":
            return self.bce(pred, target)

        elif self.loss_name == "weighted_bce":
            return self.weighted_bce(pred, target)

        elif self.loss_name == "tversky":
            return self.tversky(pred, target)

        elif self.loss_name == "bce_dice":
            return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)

        elif self.loss_name == "weighted_bce_dice":
            return 0.4 * self.weighted_bce(pred, target) + 0.6 * self.dice(pred, target)

        elif self.loss_name == "bce_tversky":
            return 0.4 * self.bce(pred, target) + 0.6 * self.tversky(pred, target)

        elif self.loss_name == "dice_tversky":
            return 0.5 * self.dice(pred, target) + 0.5 * self.tversky(pred, target)

        else:
            raise ValueError(f"Unknown loss name: {self.loss_name}")
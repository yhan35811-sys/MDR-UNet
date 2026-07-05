

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1. 参数统计工具
# =========================================================

def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# 2. 深度可分离卷积：DSConv
# =========================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution，深度可分离卷积。

    普通 3×3 卷积参数量：
        3 × 3 × Cin × Cout

    深度可分离卷积参数量：
        3 × 3 × Cin + Cin × Cout

    作用：
        用极少参数完成局部纹理提取，。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, use_act=True):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        # Depthwise Conv：每个通道单独做 3×3 卷积
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )

        # Pointwise Conv：1×1 卷积负责通道融合
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU() if use_act else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DSConvBlock(nn.Module):
    """
    轻量残差卷积块：DSConv + DSConv + shortcut。

    输入：
        B × Cin × H × W

    输出：
        B × Cout × H/s × W/s

    作用：
        1. 用深度可分离卷积降低参数量；
        2. 用残差连接保持梯度稳定；
        3. stride=2 时完成下采样。
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = DepthwiseSeparableConv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            use_act=True
        )

        self.conv2 = DepthwiseSeparableConv(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            use_act=False
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # 如果通道数或尺寸变化，需要用 1×1 shortcut 对齐
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)

        out = out + identity
        out = self.act(out)

        return out


# =========================================================
# 3. 轻量通道注意力：ECA
# =========================================================

class ECALayer(nn.Module):
    """
    Efficient Channel Attention，轻量通道注意力。

    原来的 ChannelAttention 使用 MLP：
        C -> C/r -> C

    这样虽然有效，但参数仍然会随通道数增长。

    ECA 不使用降维 MLP，只用一个 1D 卷积建模邻近通道关系，参数极少。

    作用：
        关注“哪些通道更重要”。
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B × C × H × W
        y = self.avg_pool(x)                    # B × C × 1 × 1
        y = y.squeeze(-1).transpose(-1, -2)     # B × 1 × C
        y = self.conv1d(y)                      # B × 1 × C
        y = y.transpose(-1, -2).unsqueeze(-1)   # B × C × 1 × 1

        weight = self.sigmoid(y)
        return weight


# =========================================================
# 4. 轻量空间注意力：Spatial Attention
# =========================================================

class LiteSpatialAttention(nn.Module):
    """
    轻量空间注意力。

    作用：
        关注“病灶在哪里”。

    做法：
        1. 沿通道维度计算 average map 和 max map；
        2. 拼接为 2 通道；
        3. 用 3×3 卷积生成空间权重图。

    这里使用 3×3，而不是原 CBAM 常用的 7×7，
    是为了进一步降低参数和计算量。
    """

    def __init__(self, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)      # B × 1 × H × W
        max_map, _ = torch.max(x, dim=1, keepdim=True)    # B × 1 × H × W

        att = torch.cat([avg_map, max_map], dim=1)        # B × 2 × H × W
        weight = self.conv(att)                           # B × 1 × H × W

        return weight


# =========================================================
# 5. 边界感知轻量双注意力模块：B-LDAM
# =========================================================

class EdgeAttention(nn.Module):
    """
    边界注意力分支：
    通过高频差分提取伤口边缘信息，减少背景误分，提高边界 Dice。
    """

    def __init__(self, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        edge = torch.abs(x - smooth)

        avg_map = torch.mean(edge, dim=1, keepdim=True)
        max_map, _ = torch.max(edge, dim=1, keepdim=True)

        edge_map = torch.cat([avg_map, max_map], dim=1)
        return self.conv(edge_map)


class LDAM(nn.Module):
    """
    改进版 LDAM：
        Channel Attention + Spatial Attention + Edge Attention

    mode:
        'dual'    : 通道注意力 + 空间注意力 + 边界注意力
        'channel' : 仅通道注意力
        'spatial' : 仅空间注意力
        'none'    : 不使用注意力
    """

    def __init__(self, channels, mode='dual'):
        super().__init__()

        self.mode = mode

        self.channel_att = ECALayer(channels)

        # 原来是 3×3，这里改为 5×5，提升空间感受野，参数增加很少
        self.spatial_att = LiteSpatialAttention(kernel_size=5)

        # 新增边界注意力
        self.edge_att = EdgeAttention(kernel_size=3)

        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.edge_gamma = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        if self.mode == 'none':
            return x

        out = x

        if self.mode in ['channel', 'dual']:
            out = out * self.channel_att(out)

        if self.mode in ['spatial', 'dual']:
            out = out * self.spatial_att(out)

        if self.mode == 'dual':
            edge_out = x * self.edge_att(x)
            return x + self.gamma * out + self.edge_gamma * edge_out

        return x + self.gamma * out


# =========================================================
# 6. Tiny Mamba Bottleneck，
# =========================================================

class TinyMambaBlock(nn.Module):
    """
    Tiny Mamba-style 2D Selective State-Space Block。

    这是一个轻量 Mamba/SSM 风格模块，不依赖 mamba_ssm，
    适合直接放进 Tiny.py 跑。

    输入：
        B × C × H × W

    步骤：
        1. 将二维特征展平成序列：
           B × C × H × W -> B × L × C，L=H×W；

        2. LayerNorm 稳定序列特征；

        3. in_proj 得到 u 和 z 两个分支；

        4. u 分支经过 depthwise conv 捕获局部上下文；

        5. selective_scan_lite 用 cumsum 实现轻量状态空间式长程建模；

        6. z 分支做门控；

        7. reshape 回二维特征图。

    """

    def __init__(self, dim, expand=1, dropout=0.0):
        super().__init__()

        inner_dim = dim * expand

        self.dim = dim
        self.inner_dim = inner_dim

        self.norm = nn.LayerNorm(dim)

        # 输入投影，一半用于状态空间建模 u，一半用于门控 z
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)

        # 局部混合，保持 Mamba 中短卷积建模局部上下文的思想
        self.dwconv = nn.Conv2d(
            inner_dim,
            inner_dim,
            kernel_size=3,
            padding=1,
            groups=inner_dim,
            bias=False
        )

        # 选择性状态空间的三个轻量投影
        self.dt_proj = nn.Linear(inner_dim, inner_dim, bias=True)
        self.b_proj = nn.Linear(inner_dim, inner_dim, bias=False)
        self.c_proj = nn.Linear(inner_dim, inner_dim, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def selective_scan_lite(self, u):
        """
        轻量选择性扫描。

        u:
            B × L × C

        这里用前向累计 + 后向累计近似长程依赖建模：

            forward scan:
                建模从左到右 / 从前到后的上下文；

            backward scan:
                建模从右到左 / 从后到前的上下文。


        """

        delta = F.softplus(self.dt_proj(u))        # B × L × C，动态步长
        b_gate = torch.sigmoid(self.b_proj(u))     # B × L × C，输入选择门
        c_gate = torch.sigmoid(self.c_proj(u))     # B × L × C，输出选择门

        weight = delta * b_gate                    # B × L × C

        # forward scan
        numerator = torch.cumsum(weight * u, dim=1)
        denominator = torch.cumsum(weight, dim=1).clamp_min(1e-6)
        y_forward = c_gate * numerator / denominator

        # backward scan
        u_rev = torch.flip(u, dims=[1])
        weight_rev = torch.flip(weight, dims=[1])
        c_gate_rev = torch.flip(c_gate, dims=[1])

        numerator_rev = torch.cumsum(weight_rev * u_rev, dim=1)
        denominator_rev = torch.cumsum(weight_rev, dim=1).clamp_min(1e-6)
        y_backward = c_gate_rev * numerator_rev / denominator_rev
        y_backward = torch.flip(y_backward, dims=[1])

        y = 0.5 * (y_forward + y_backward)

        return y

    def forward(self, x):
        # x: B × C × H × W
        B, C, H, W = x.shape

        # 2D -> sequence
        seq = x.flatten(2).transpose(1, 2)         # B × L × C
        residual = seq

        seq = self.norm(seq)

        # u: 状态空间分支
        # z: 门控分支
        u, z = self.in_proj(seq).chunk(2, dim=-1)  # B × L × inner_dim

        # depthwise conv 在二维空间做局部混合
        u_2d = u.transpose(1, 2).reshape(B, self.inner_dim, H, W)
        u_2d = self.dwconv(u_2d)
        u_2d = self.act(u_2d)
        u = u_2d.flatten(2).transpose(1, 2)        # B × L × inner_dim

        # 选择性状态空间扫描，建模全局依赖
        y = self.selective_scan_lite(u)

        # Mamba 常见门控形式
        y = y * self.act(z)

        # 输出投影 + 残差
        y = self.out_proj(y)
        y = self.dropout(y)
        y = y + residual

        # sequence -> 2D
        y = y.transpose(1, 2).reshape(B, C, H, W)

        return y




# =========================================================
# 8. 解码器上采样模块
# =========================================================

class UpBlock(nn.Module):
    """
    轻量解码器模块。

    这里：
        Bilinear Upsample + L-DAM skip attention + DSConvBlock

    输入：
        x:
            深层特征

        skip:
            编码器浅层特征

    输出：
        融合后的解码特征
    """

    def __init__(self, in_channels, skip_channels, out_channels, attention_mode='dual', dropout=0.0):
        super().__init__()

        self.skip_attention = LDAM(skip_channels, mode=attention_mode)

        self.fuse = DSConvBlock(
            in_channels + skip_channels,
            out_channels,
            stride=1,
            dropout=dropout
        )

    def forward(self, x, skip):
        # 双线性上采样到 skip 的尺寸
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 对 skip connection 做轻量双注意力，抑制背景噪声
        skip = self.skip_attention(skip)

        # 拼接并融合
        x = torch.cat([skip, x], dim=1)
        x = self.fuse(x)

        return x


# =========================================================
# 9. 主模型：MDRUNet
# =========================================================

class MDRUNet(nn.Module):
    """
    轻量部署版分割模型。

    整体结构：

        Input
          ↓
        Stem DSConv
          ↓
        Tiny Encoder: DSConvBlock × 4
          ↓
        Tiny Mamba Bottleneck
          ↓
        Decoder: Bilinear Up + L-DAM + DSConvBlock × 4
          ↓
        1×1 Conv 输出 mask

    默认通道：
        channels=(16, 32, 48, 64)

    默认参数量：
        约 0.075M，可满足 0.05M–0.10M。
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        channels=(16, 32, 48, 64),
        attention_mode='dual',
        mamba_type='lite',
        dropout=0.0
    ):
        super().__init__()

        c1, c2, c3, c4 = channels

        # -------------------------
        # Encoder
        # -------------------------

        # stem 不降采样，保留原图细节
        self.stem = DSConvBlock(
            in_channels,
            c1,
            stride=1,
            dropout=dropout
        )   # H × W

        # encoder 逐级下采样
        self.enc1 = DSConvBlock(
            c1,
            c1,
            stride=2,
            dropout=dropout
        )   # H/2 × W/2

        self.enc2 = DSConvBlock(
            c1,
            c2,
            stride=2,
            dropout=dropout
        )   # H/4 × W/4

        self.enc3 = DSConvBlock(
            c2,
            c3,
            stride=2,
            dropout=dropout
        )   # H/8 × W/8

        self.enc4 = DSConvBlock(
            c3,
            c4,
            stride=2,
            dropout=dropout
        )   # H/16 × W/16

        # -------------------------
        # Tiny Mamba Bottleneck
        # -------------------------

        if mamba_type == 'lite':
            # Mamba 负责全局建模，后接 DSConvBlock 做局部边界细化
            self.bottleneck = nn.Sequential(
                TinyMambaBlock(
                    c4,
                    expand=1,
                    dropout=dropout
                ),
                DSConvBlock(
                    c4,
                    c4,
                    stride=1,
                    dropout=dropout
                )
            )
        else:
            raise ValueError("mamba_type 只能是 'lite' 或 'official'")

        # -------------------------
        # Decoder
        # -------------------------

        self.dec4 = UpBlock(
            c4,
            c3,
            c3,
            attention_mode=attention_mode,
            dropout=dropout
        )   # H/8

        self.dec3 = UpBlock(
            c3,
            c2,
            c2,
            attention_mode=attention_mode,
            dropout=dropout
        )   # H/4

        self.dec2 = UpBlock(
            c2,
            c1,
            c1,
            attention_mode=attention_mode,
            dropout=dropout
        )   # H/2

        self.dec1 = UpBlock(
            c1,
            c1,
            c1,
            attention_mode=attention_mode,
            dropout=dropout
        )   # H

        # -------------------------
        # Output refinement
        # -------------------------

        self.refine = nn.Sequential(
            DSConvBlock(
                c1,
                c1,
                stride=1,
                dropout=dropout
            ),
            LDAM(
                c1,
                mode=attention_mode
            )
        )

        # -------------------------
        # Output head
        # -------------------------

        self.head = nn.Sequential(
            DepthwiseSeparableConv(
                c1,
                c1,
                kernel_size=3,
                stride=1,
                use_act=True
            ),
            nn.Conv2d(
                c1,
                n_classes,
                kernel_size=1
            )
        )

    def forward(self, x):
        # Encoder
        s0 = self.stem(x)        # B × c1 × H    × W
        e1 = self.enc1(s0)       # B × c1 × H/2  × W/2
        e2 = self.enc2(e1)       # B × c2 × H/4  × W/4
        e3 = self.enc3(e2)       # B × c3 × H/8  × W/8
        e4 = self.enc4(e3)       # B × c4 × H/16 × W/16

        # Mamba bottleneck
        b = self.bottleneck(e4)  # B × c4 × H/16 × W/16

        # Decoder with L-DAM skip fusion
        d4 = self.dec4(b, e3)    # B × c3 × H/8 × W/8
        d3 = self.dec3(d4, e2)   # B × c2 × H/4 × W/4
        d2 = self.dec2(d3, e1)   # B × c1 × H/2 × W/2
        d1 = self.dec1(d2, s0)   # B × c1 × H   × W

        # 输出前再细化一次边界和病灶区域
        d1 = self.refine(d1)

        # Output logits
        # 训练时建议直接喂给 BCEWithLogitsLoss，不要在这里 sigmoid
        out = self.head(d1)      # B × n_classes × H × W

        return out


# =========================================================
# 10. 模型选择区：和你原来的 get_model() 对齐
# =========================================================

def get_model():
    """
    直接在训练代码里调用：

        model = get_model()

    """

    n_classes = 1

    model = MDRUNet(
        in_channels=3,
        n_classes=n_classes,
        channels=(48, 96, 144, 192),
        attention_mode='dual',         # dual / channel / spatial / none
        mamba_type='lite',             # lite / official
        dropout=0.00
    )

    return model


# =========================================================
# 11. 自测代码
# =========================================================

if __name__ == "__main__":
    model = get_model()

    params = count_parameters(model)

    print(model)
    print(f"Trainable params: {params:,} = {params / 1e6:.4f} M")

    # 用小图测试速度；实际训练 512×512 也可以
    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
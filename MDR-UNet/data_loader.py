import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config


class DFUDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        self.images = [
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        self.images = sorted(self.images)

        if len(self.images) == 0:
            raise RuntimeError(f"没有在图像文件夹中找到图片: {image_dir}")

    def __len__(self):
        return len(self.images)

    def find_mask_path(self, image_name):
        """
        根据 image 文件名寻找对应 mask。
        支持：
        image: xxx.jpg / xxx.jpeg / xxx.png
        mask : xxx.png / xxx.jpg / xxx.jpeg / xxx.bmp / xxx.tif / xxx.tiff
        """
        name, _ = os.path.splitext(image_name)

        possible_masks = [
            name + ".png",
            name + ".jpg",
            name + ".jpeg",
            name + ".bmp",
            name + ".tif",
            name + ".tiff",
        ]

        for mask_name in possible_masks:
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                return mask_path

        raise FileNotFoundError(
            f"找不到对应 mask: image={image_name}, mask_dir={self.mask_dir}"
        )

    def __getitem__(self, index):
        image_name = self.images[index]

        img_path = os.path.join(self.image_dir, image_name)
        mask_path = self.find_mask_path(image_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"图像读取失败: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise FileNotFoundError(f"mask 读取失败: {mask_path}")

        # mask 二值化：背景=0，伤口=1
        mask = np.where(mask > 127, 1.0, 0.0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


def get_train_transform():
    """
    训练集增强：
    适合 DFU 慢性伤口分割：
    1. 几何变化：平移、缩放、旋转、翻转
    2. 轻微形变：ElasticTransform / GridDistortion
    3. 光照颜色：Brightness、Contrast、HSV、Gamma、CLAHE
    4. 图像质量：噪声、模糊
    """

    train_transform = A.Compose([
        A.Resize(
            height=Config.IMAGE_HEIGHT,
            width=Config.IMAGE_WIDTH,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),

        # 替代 ShiftScaleRotate，避免 warning
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.05, 0.05),
            rotate=(-25, 25),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            fill=0,
            fill_mask=0,
            p=0.6
        ),

        A.HorizontalFlip(p=0.5),

        # 伤口图像不建议强烈上下翻转，概率低一点
        A.VerticalFlip(p=0.05),

        # 轻微形变，模拟伤口边界变化
        A.OneOf([
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.15,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0
            ),
        ], p=0.15),

        # 光照和颜色增强
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=1.0
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=1.0
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=1.0
            ),
        ], p=0.7),

        # 图像质量扰动：噪声、模糊
        A.OneOf([
            A.GaussNoise(
                std_range=(0.02, 0.08),
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=1.0
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=1.0
            ),
            A.MotionBlur(
                blur_limit=3,
                p=1.0
            ),
        ], p=0.25),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ])

    return train_transform


def get_val_transform():
    """
    验证集不做随机增强。
    只做 Resize + Normalize + ToTensor。
    """

    val_transform = A.Compose([
        A.Resize(
            height=Config.IMAGE_HEIGHT,
            width=Config.IMAGE_WIDTH,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),

        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),

        ToTensorV2(),
    ])

    return val_transform


def get_loaders():
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_ds = DFUDataset(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MASK_DIR,
        transform=train_transform
    )

    val_ds = DFUDataset(
        Config.VAL_IMG_DIR,
        Config.VAL_MASK_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    return train_loader, val_loader
import torch
import os


class Config:
    # 路径设置
    # 请确保路径正确
    DATA_ROOT = r"D:\my_code\DFU_DL_Test\data\Foot Ulcer Segmentation Challenge"

    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train/images")
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train/labels")

    VAL_IMG_DIR = os.path.join(DATA_ROOT, "test/images")
    VAL_MASK_DIR = os.path.join(DATA_ROOT, "test/labels")

    RESULTS_DIR = "./results"
    CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

    # 训练过程中的可视化图
    VIS_TRAIN_DIR = os.path.join(RESULTS_DIR, "vis_train_progress")
    # 对比的图
    VIS_RESULT_DIR = os.path.join(RESULTS_DIR, "vis_final_comparison")

    for d in [CHECKPOINT_DIR, PLOT_DIR, VIS_TRAIN_DIR, VIS_RESULT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 关键设置：固定的5张图索引
    # 我们将始终使用这5张图来进行可视化对比
    #1,5,10,15  FUSC2021
    #1,5,10,20  DFU2022
    # 你可以根据数据集大小修改这些数字，确保它们在验证集范围内
    VIS_INDICES = [1, 2, 5, 6]

    # 训练参数
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 200
    NUM_WORKERS = 0
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512

    # 模型名称
    MODEL_NAME = "MDR_UNet"

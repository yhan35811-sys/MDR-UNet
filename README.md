# MDR-UNet

## A Global–Boundary Collaborative Lightweight Network for Diabetic Foot Ulcer Segmentation



This repository provides the official PyTorch implementation of **MDR-UNet**, a lightweight segmentation network for diabetic foot ulcer (DFU) images.

High-accuracy DFU segmentation methods commonly rely on large backbones or computationally expensive feature-enhancement modules. In contrast, conventional lightweight networks often lose segmentation accuracy because of insufficient local-detail preservation, global-context modeling, and boundary discrimination. MDR-UNet is designed to alleviate this accuracy–efficiency dilemma through coordinated local, global, and boundary-aware feature modeling.

<p align="center">
  <img src="https://github.com/user-attachments/assets/094c33fe-43b9-4c53-b8ab-c3de07eb21d8"
       alt="Overall architecture of MDR-UNet"
       width="95%">
</p>

<p align="center">
  <b>Fig. 1. Overall architecture of MDR-UNet.</b>
</p>
---

### Expected directory structure

```text
data/
├── FUSC2021/
│   ├── images/
│   └── masks/
└── DFUC2022/
    ├── images/
    └── masks/
```

Update the dataset paths in `config.py` before training or evaluation. Image and mask filenames must correspond one-to-one.

---

## Repository Structure

```text
MDR-UNet/
├── assets/                  # Architecture and qualitative figures
├── models/
│   └── MDR_UNet.py          # MDR-UNet architecture
├── config.py                # Paths and experimental settings
├── data_loader.py           # Data loading and augmentation
├── losses.py                # BCE and Tversky-based losses
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
├── utils.py                 # Metrics, checkpoints, and utilities
├── requirements.txt         # Python dependencies
├── README.md
└── LICENSE
```


## Installation

1. Clone the repository to your local machine:

   Bash

   ```
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
   ```

2. Create a virtual environment (optional but recommended) and install the required dependencies:

   Bash

   ```
   pip install -r requirements.txt
   ```

   *Note: Ensure your environment supports PyTorch with CUDA for GPU acceleration.*

## Usage

### 1. Data Preparation

Place your DFU image dataset into the `data/` directory. Ensure that the paths in `config.py` correctly point to your training, validation, and testing splits.

### 2. Configuration

Adjust training hyperparameters (learning rate, batch size, epochs) and model settings directly in `config.py`.

### 3. Training

To start training the segmentation model, run:

Bash

```
python train.py
```

The script will automatically save the best model weights and training logs into the `src/results/` directory.

### 4. Evaluation

To evaluate the trained model on your test dataset and compute metrics like Dice, run:

Bash

```
python evaluate.py
```



## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Paper

**MDR-UNet: A Global–Boundary Collaborative Lightweight Segmentation Network for Diabetic Foot Ulcer Analysis**

The manuscript is currently under review. The publication link, DOI, and final citation metadata will be added after publication.


## Licence

The source code should be released with an explicit open-source licence. A permissive licence such as the MIT License may be used by adding a complete `LICENSE` file to the repository.

The dataset licences are independent of the source-code licence. Users are responsible for obtaining dataset access and complying with the original providers' terms.

## Contact

For questions about the manuscript, implementation, or experimental protocol, please contact:

**Yawu Zhao**  
Corresponding author  
Email: `zhaoyawu9608@163.com`

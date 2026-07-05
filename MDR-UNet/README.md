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

Place your DFU image dataset into the `data/` directory. Ensure that the paths in `src/config.py` correctly point to your training, validation, and testing splits.

### 2. Configuration

Adjust training hyperparameters (learning rate, batch size, epochs) and model settings directly in `src/config.py`.

### 3. Training

To start training the segmentation model, run:

Bash

```
python src/train.py
```

The script will automatically save the best model weights and training logs into the `src/results/` directory.

### 4. Evaluation

To evaluate the trained model on your test dataset and compute metrics like Dice and HD95, run:

Bash

```
python src/evaluate.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
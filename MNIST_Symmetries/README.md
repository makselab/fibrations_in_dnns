# MLP Fibration Symmetry Analysis

Research into structural symmetries of trained neural networks — specifically **fibrations**, **opfibrations**, and **coverings** in the weight graph — and their use for principled network compression.

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Hardware & Environment](#hardware--environment)

---

## Installation

### Requirements

- Python 3.8
- CUDA 11.3 compatible GPU (tested on NVIDIA Quadro RTX 6000)

### 1. Clone the repository

```bash
git clone https://github.com/OsvaVelarde/fibrations_in_dnns.git
cd fibrations_in_dnns/MNIST_Symmetries
```

### 2. Create a virtual environment (recommended)

```bash
python3.8 -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with CUDA support

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure paths

Edit `cfgfiles/exp_01.json` and set `dataset_path` to the directory where MNIST will be downloaded:

```json
"data": {
    "dataset_path": "/path/to/your/data/"
}
```

The dataset will be downloaded automatically on the first run of `training.py`.

### 6. Verify installation

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
CUDA available: True
```

---

## Dataset

The experiments use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset of handwritten digits.

| Property | Value |
|----------|-------|
| **Input size** | 28 × 28 grayscale images → flattened to 784-dimensional vector |
| **Classes** | 10 (digits 0–9) |
| **Training samples** | 60,000 |
| **Test samples** | 10,000 |

### Accessing the Dataset

The dataset is downloaded automatically by `torchvision` on the first run of `training.py`:

```python
from torchvision.datasets import MNIST
MNIST(root=dataPATH, train=True, download=True)
```

A local copy of the raw files is also included in `MNIST/raw/`.

### Preprocessing

The only preprocessing applied is `torchvision.transforms.ToTensor()`, which:
- Converts PIL images to PyTorch tensors
- Rescales pixel values from **[0, 255]** to **[0.0, 1.0]**

Images are then flattened from (1, 28, 28) to a 784-dimensional vector before being fed to the MLP:

```python
images = images.view(-1, 784)
```

No normalization, augmentation, or additional transforms are applied.

### Data Splitting

The standard MNIST split is used without modification:

| Split | Samples | Shuffle | Batch size |
|-------|---------|---------|------------|
| Train | 60,000  | Yes     | 100        |
| Test  | 10,000  | No      | 100        |

No validation set is used. No custom splitting is performed.

### Training Protocol

Training consists of a **single pass** over the training set (1 epoch = 600 batches of 100 samples). A model checkpoint is saved after every batch, resulting in 600 checkpoints (`model_batch_0.pth` to `model_batch_599.pth`). Throughout the paper, *batch index* and *epoch* are used interchangeably to refer to these 600 training steps.

The test set is used exclusively for evaluation — never for training or threshold selection.

---

## Model

- **Architecture:** `MLP` — fully-connected feedforward network
- **Layer sizes:** `[784, 500, 500, 500, 10]` (input → 3 hidden layers → output)
- **Activation:** ReLU after each hidden layer; no activation on output layer
- **Parameters:** 898,510
- **Framework:** PyTorch
- **Optimizer:** Adam (lr = 0.001)
- **Loss function:** Cross-entropy
- **Batch size:** 100

### Symmetry & Compression Methods

The `MLP` class extends a standard classifier with symmetry-based operations computed after training:

| Method | Purpose |
|--------|---------|
| `fibration_coloring` | Groups nodes by forward (fibration) symmetry per layer |
| `opfibration_coloring` | Groups nodes by backward (opfibration) symmetry per layer |
| `covering_coloring` | Combines fibration + opfibration into a joint covering symmetry |
| `collapse_version` | Builds a smaller MLP by merging nodes that share a color |
| `ablation_version` | Builds a smaller MLP by randomly removing nodes (baseline) |
| `pruning_version` | Builds a smaller MLP via structured L1-norm magnitude pruning (baseline) |

### Notes

- Symmetry detection depends on clustering distance thresholds, which are set per layer.
- `ablation_version` uses random node selection (`randperm`); results should be averaged over multiple runs.


---

## Hardware & Environment

### Hardware

| Component | Specification |
|-----------|--------------|
| **CPU** | Intel Xeon W-2245 @ 3.90 GHz (8 cores / 16 threads, boost up to 4.7 GHz) |
| **RAM** | 128 GB |
| **GPU** | NVIDIA Quadro RTX 6000 (24 GB VRAM) |
| **Storage** | NVMe SSD |

### Operating System

Ubuntu 20.04.6 LTS

### Dependencies

| Package | Version |
|---------|---------|
| **Python** | 3.8.10 |
| **CUDA** (driver) | 11.4 |
| **CUDA** (PyTorch) | 11.3 |
| **PyTorch** | 1.12.1+cu113 |
| **torchvision** | 0.13.1+cu113 |
| **NumPy** | 1.23.3 |
| **matplotlib** | 3.7.5 |
| **pandas** | 2.0.2 |
| **scikit-learn** | 1.2.2 |
| **SciPy** | 1.10.1 |


----------------------------------------

# MLP Fibration Symmetry Analysis

## Initial Setup

Define the correct paths in the various scripts for saving the dataset (`dataPATH`) and the results (`resultsPATH`):

- **`dataPATH`**: Directory where the MNIST dataset will be stored.
- **`resultsPATH`**: Directory where experiment results will be saved. Must include the following subfolders:
  - `clustering`
  - `collapse`
  - `coloring`
  - `plots`
  - `training`

---

## Script Descriptions

### Core Functionality
- **`coloring.py`**: Contains functions for computing fibrations and opfibrations in linear layers.
- **`model.py`**: Defines the MLP model (a 3-hidden-layer neural network). The `MLP` class (a `nn.Module`) includes methods for:
  - Computing fibrations, opfibrations, and coverings using the network's current parameters.

---

### Training & Activity Generation
- **`training.py`**: Trains the MLP model on MNIST classification.
  - Model weights are saved in `resultsPATH/training/weight_batch_bb.pth` (where `bb = 0, ..., 599`).
  - Training accuracy is saved as `resultsPATH/training/accuracy.pth` (a 1D tensor with 600 elements).

- **`generate_activity.py`**: Generates node activity using:
  1. Samples from the MNIST evaluation subset (10,000 samples).
  2. Randomly generated samples (200 samples).
  - Results saved as:
    - `activity_batch_599.pth`
    - `activity_random_input_batch_599.pth`

---

### Symmetry Analysis
- **`fibration_coloring.py`**: Computes fibrations for the MLP at training steps `bb = 0, ..., 599`.
  - **`threshold`** parameter controls fiber granularity:
    - `threshold = 0` → Each node in its own fiber (no symmetries).
    - `threshold = 2` → All nodes in one fiber (full symmetry).
  - Results saved in `coloring/fibration_batch_bb.pth` for thresholds `(0, 0.01, ..., 1.00)`.

- **`synchronization_clusters.py`**: Computes synchronization clusters per layer using random-input activity.
  - Requires `epsilon` parameter (100 values between `0` and `2`).
  - Results saved in `clustering/clusters_batch_599.pth` for step `bb = 599`.

---

### Collapse & Evaluation
- **`collapse_during_training.py`**, **`collapse_post_training.py`**:  
  Compare original vs. collapsed model performance (using fibration symmetries) at different training stages/thresholds.
- **`plt_during_training.py`**, **`plt_post_training.py`**:  
  Visualize results and evaluate network size reduction.

---

### Matching Synchronization & Fibrations
- **`matching.py`**: Establishes relationship between:
  - `epsilon` (synchronization clusters)
  - `threshold` (fibration symmetries)

 ---

### Reproducing the results
To reproduce the results shown in the `plots/` directory, execute the following scripts (src) in order (python ...):
- training.py
- generate_activity.py
- fibration_coloring.py
- synchronization_clusters.py
- collapse_during_training.py
- collapse_post_training.py
- plt_during_training.py
- plt_post_training.py
- matching.py


----------------

train_dir (OMV5TB)

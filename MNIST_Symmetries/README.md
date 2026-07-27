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


---

## How to Run

All scripts are run from the root of the repository. The experiment ID (e.g., `01`) is passed as the first argument to each shell script.

### 1. Training, compression & symmetry tracking

```bash
bash main.sh 01
```

Runs in order:
- `src/training.py` — trains the MLP, saves 600 checkpoints
- `src/compression.py` — builds collapsed, pruned and ablated models
- `src/evaluation_compressed.py` — evaluates all compressed models on the test set
- `src/symmetry_vs_time.py` — tracks fibration/opfibration symmetry across training (3 threshold configurations)

### 2. Optimal curve search & Pareto frontier

```bash
bash optimal_curves.sh 01
```

Runs in order:
- `src/search.py` — finds optimal fibration thresholds per opfibration level (11 curves)
- `src/full_search.py` — full grid search over all threshold combinations
- `src/full_search_pareto.py` — computes Pareto frontier from full grid (downloads `Full_Grid_dL_vs_thr.csv` from Zenodo if not present)
- `src/evaluation_pareto_frontier.py` — evaluates collapsed models on the Pareto frontier on the test set

### 3. Synchronization analysis

```bash
bash synchronization.sh 01
```

Runs in order:
- `src/synchronization/generate_activity.py`
- `src/synchronization/generate_activity_mean_class.py`
- `src/synchronization/clusters.py`
- `src/synchronization/clusters_per_class.py`
- `src/synchronization/generate_fibration_coloring.py`
- `src/synchronization/matching.py`

### 4. Plots

```bash
bash plots.sh 01
```

Generates all figures from the results of steps 1–3.

### Optional: continue training from a checkpoint

```bash
python3 src/re-training-check.py -exp_name exp_01 -PATHtrain ./train_dir/ -PATHcfg ./cfgfiles/
```

Loads the latest checkpoint (compressed or uncompressed) and continues training for one additional epoch.

---

## Model Card

### Intended Use

- **Primary intended use:** Research into structural symmetries of trained neural networks — specifically **fibrations**, **opfibrations**, and **coverings** in the weight graph — and their use for principled network compression.
- **Primary intended users:** Researchers studying neural network symmetry/compression (this lab, reviewers/readers of the associated Nature Machine Intelligence submission).
- **Out-of-scope uses:** Not intended as a production MNIST classifier; the architecture is a controlled research vehicle for studying symmetry structure, not optimized for state-of-the-art accuracy.

### Evaluation

**Base model accuracy (epoch 599, test set)**

| Metric | Value |
|--------|-------|
| Accuracy | **96.78 %** |

**Compression evaluation (Pareto frontier — `Ev_pareto_frontier_599.csv`)**

| Retained parameters | Retained nodes | Accuracy |
|--------------------|---------------|----------|
| ~100 % (898,510)  | ~100 % (1,500) | 96.78 % |
| ~97.5 % (876,257) | ~98 % (1,470)  | 96.78 % |
| ~66.4 % (596,855) | ~74 % (1,111)  | 96.55 % |
| ~49.7 % (446,000) | ~66 % (990)    | 96.33 % |
| ~9.9 % (89,200)   | ~17 % (255)    | 82.10 % |
| ~0.56 % (5,003)   | ~2 % (26)      | 22.81 % |

**Clustering method**

- **Algorithm:** `AgglomerativeClustering` (scikit-learn) with **average linkage** and precomputed cosine distance matrix.
- **Distance threshold range:** searched over `[0.4, 1.0]` (11 values per layer, 6 layers → full grid of 1,771,561 combinations in `full_search.py`).

**Distance thresholds used (symmetry vs. time analysis)**

| Configuration | fib thresholds (L1, L2, L3) | opfib thresholds (L1, L2, L3) |
|---------------|-----------------------------|-------------------------------|
| A | 0.75, 0.65, 0.15 | 1.0, 1.0, 1.0 |
| B | 0.80, 0.80, 0.80 | 1.0, 1.0, 1.0 |
| C | 1.50, 1.50, 1.50 | 0.8, 0.8, 0.3 |

### Limitations

- Architecture and hyperparameters are tuned for a controlled MNIST symmetry study, not for benchmark-competitive performance.
- Symmetry detection depends on clustering distance thresholds set per layer — results may be sensitive to this choice.
- `ablation_version` uses random node selection (`randperm`) with no fixed seed; results should be averaged over multiple runs for a fair baseline comparison.
- Findings on MNIST/MLP may not generalize directly to convolutional or larger-scale architectures without further validation.

### Ethical Considerations

- Standard MNIST digit classification; no personally identifiable or sensitive data involved.
- No foreseeable direct societal risk; this is foundational research into network structure and compression methods.

---

## Results (exp_01)

### pareto_frontier.csv
**Rows:** 4893 | **Generated by:** `src/full_search.py` + `src/full_search_pareto.py`

Full grid search over fibration and opfibration thresholds (per layer). Each row is a threshold combination with the resulting structural loss (`dL`) and parameter reduction of the collapsed model.

Columns: `fib_0, fib_1, fib_2, opf_0, opf_1, opf_2, dL, reduction_pars_coll`

### compression_results_thr_599.csv
**Rows:** 93 | **Generated by:** `src/compression.py`

Structural metrics (sizes and reductions) for collapsed, pruned, and 10 ablated models at specific threshold combinations (epoch 599). Does **not** include accuracy.

Columns: `thr_fib_*, thr_opfib_*, num_nodes_coll, reduction_nodes, num_colors_l1/l2/l3, num_params_coll, reduction_pars_coll, num_L{1,2,3}_pruned, num_params_pruned, reduction_pars_pruned, num_L{1,2,3}_abl_{0..9}, num_params_abl_{0..9}, reduction_pars_abl_{0..9}`

### Evaluation_CompressedModels_epoch_599.csv
**Rows:** 93 | **Generated by:** `src/evaluation_compressed.py`

Accuracy and loss for collapsed, pruned, and 10 ablated models. Same threshold combinations as `compression_results_thr_599.csv`.

Columns: `thr_fib_*, thr_opfib_*, acc_coll, acc_pruned, loss_coll, loss_pruned, acc_abl_{0..9}, loss_abl_{0..9}`

### Ev_pareto_frontier_599.csv
**Rows:** 4893 | **Generated by:** `src/evaluation_pareto_frontier.py`

Same grid as `pareto_frontier.csv` but extended with accuracy and structural metrics of the collapsed model (epoch 599).

Columns: `fib_0, fib_1, fib_2, opfib_0, opfib_1, opfib_2, num_nodes_coll, reduction_nodes, num_colors_l1/l2/l3, num_params_coll, reduction_pars_coll, acc_coll, loss_coll`

### Symmetries_thrs_*.csv (3 files)
**Rows:** 600 | **Generated by:** `src/symmetry_vs_time.py`

Symmetry metrics across training epochs at a fixed threshold configuration. One file per threshold setting: `[0.75, 0.65, 0.15, 1.0, 1.0, 1.0]`, `[0.8, 0.8, 0.8, 1.0, 1.0, 1.0]`, `[1.5, 1.5, 1.5, 0.8, 0.8, 0.3]`.

Columns: `epoch, num_nodes_coll, num_params_coll, reduction_pars_coll, reduction_nodes, num_colors_l0, num_colors_l1, num_colors_l2`

### optimal_curves/opfib_{thr}.csv (11 files)
**Rows:** ~31 | **Generated by:** `src/search.py`

One file per opfibration threshold (0.45 to 1.0, step 0.05). Each file contains the optimal fibration threshold combination that minimizes `dL` at each level of parameter reduction.

Columns: `dL_thr, reduction_pars_coll, dL_cov, thr_fib_0, thr_fib_1, thr_fib_2`

### Full_Grid_dL_vs_thr.txt
Pointer to the full grid CSV file (257 MB) hosted on Zenodo. Downloaded automatically by `src/full_search_pareto.py` if the CSV is not present locally.

[https://zenodo.org/records/21499971/files/Full_Grid_dL_vs_thr.csv](https://zenodo.org/records/21499971/files/Full_Grid_dL_vs_thr.csv)

### coloring/fibration_L1_batch_599.pth
**Generated by:** `src/synchronization/generate_fibration_coloring.py`

Fibration coloring of Layer 1 over a linspace grid of thresholds `[0, 1]`. Shape: `(num_thrs, 1 + hidden_size_L1)` — first column is the threshold, remaining columns are integer fiber labels per neuron.

### synchronization/
**Generated by:** `bash synchronization.sh`

| File | Description |
|------|-------------|
| `activity_batch_599.pth` | Neuron activations on the full test set (10,000 samples) |
| `activity_random_input_batch_599.pth` | Neuron activations on random inputs (used for clustering) |
| `activity_mean_class_batch_599.pth` | Neuron activations on the mean image of each class (10 samples) |
| `clusters_batch_599.pth` | Sync clusters over epsilon grid (random inputs) — dict with keys `L1..Ln`, `eps` |
| `num_clusters_batch_599.pth` | Number of clusters per layer per epsilon |
| `distance_matrices_batch_599.pth` | Pairwise distance matrices per layer (random inputs) |
| `clusters_class_{0..9}_batch_599.pth` | Sync clusters per class per epsilon grid |
| `num_clusters_class_{0..9}_batch_599.pth` | Number of clusters per class per epsilon |
| `distance_matrices_class_{0..9}_batch_599.pth` | Distance matrices per class |
| `mean_clusters_class_{0..9}_batch_599.pth` | Sync clusters of the mean class image per epsilon |
| `mean_num_clusters_class_{0..9}_batch_599.pth` | Number of clusters of the mean class image per epsilon |
| `distance_matrices_mean_class_{0..9}_batch_599.pth` | Distance matrices for mean class images |



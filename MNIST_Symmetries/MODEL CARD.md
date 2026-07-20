# Model Card: MLP with Fibration/Opfibration Symmetry Analysis (exp_01)

## Model Details

- **Task:** MNIST Classification (10-class handwritten digit image classification)
- **Model name / architecture:** `MLP` (fully-connected feedforward network)
- **Framework:** PyTorch
- **Layer sizes:** `[784, 500, 500, 500, 10]` (input → 3 hidden layers → output)
- **Activation function:** ReLU (applied after each hidden layer; no activation on output layer)
- **Number of parameters:** 898,510 (784→500: 392,500; 500→500: 250,500; 500→500: 250,500; 500→10: 5,010 — includes biases)
- **Version / experiment ID:** `exp_01`
- **Developed by:** Osva, MakseLab
- **License:** MIT
- **Repository / code reference:** `model.py` (custom `MLP` class), depends on internal `symmetries` and `compression_methods` packages

## Intended Use

- **Primary intended use:** Research into structural symmetries of trained neural networks — specifically **fibrations**, **opfibrations**, and **coverings** in the weight graph — and their use for principled network compression.
- **Primary intended users:** Researchers studying neural network symmetry/compression (this lab, reviewers/readers of the associated Nature Machine Intelligence submission).
- **Out-of-scope uses:** Not intended as a production MNIST classifier; the architecture is a controlled research vehicle for studying symmetry structure, not optimized for state-of-the-art accuracy.

## Model Architecture / Methodology

The `MLP` class extends a standard feedforward classifier with several symmetry- and compression-related operations computed *after* training:

| Method | Purpose |
|--------|---------|
| `fibration_coloring` | Colors nodes by clustering forward (fibration) symmetry per layer, given per-layer distance thresholds |
| `opfibration_coloring` | Colors nodes by clustering backward (opfibration) symmetry per layer |
| `covering_coloring` | Combines fibration + opfibration colorings into a joint "covering" symmetry |
| `num_colors` | Reports the number of distinct symmetry classes (colors) per layer |
| `collapse_version` | Builds a smaller MLP by collapsing nodes that share a color (lossy compression exploiting symmetry) |
| `ablation_version` | Builds a smaller MLP by randomly removing (ablating) a fixed total number of nodes, as a baseline against symmetry-based collapse |
| `pruning_version` | Builds a smaller MLP via structured L1-norm magnitude pruning (`torch.nn.utils.prune`), as a second baseline |

This supports a compare-and-contrast experimental design: **symmetry-based collapse** vs. **random ablation** vs. **magnitude pruning**, at matched compression levels.

## Training Data

- **Dataset:** MNIST (handwritten digit classification)
- **Input size:** 784 (28×28 flattened, grayscale images)
- **Number of classes:** 10 (digits 0–9)
- **Preprocessing:** `ToTensor()` only — rescales pixel values from [0, 255] to [0.0, 1.0]; images flattened to 784-dimensional vectors. No normalization or augmentation applied.
- **Train / test split:** Standard MNIST split — 60,000 training samples / 10,000 test samples. No validation set used.

## Training Procedure

- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Batch size:** 100
- **Number of epochs:** 1 (single pass over the training set)
- **Training steps:** 600 batches of 100 samples
- **Loss function:** Cross-entropy
- **Hardware:** Intel Xeon W-2245 @ 3.90 GHz, NVIDIA Quadro RTX 6000 (24 GB VRAM), 128 GB RAM, Ubuntu 20.04.6 LTS
- **Random seed:** Not fixed (no `torch.manual_seed` set in `training.py`)

## Evaluation

### Base model accuracy (epoch 599, test set)

| Metric | Value |
|--------|-------|
| Accuracy | **96.78 %** |

### Compression evaluation (Pareto frontier — `Ev_pareto_frontier_599.csv`)

Symmetry-based collapse (`collapse_version`) evaluated across the Pareto-optimal set of threshold configurations:

| Retained parameters | Retained nodes | Accuracy |
|--------------------|---------------|----------|
| ~100 % (898,510)  | ~100 % (1,500) | 96.78 % |
| ~97.5 % (876,257) | ~98 % (1,470)  | 96.78 % |
| ~66.4 % (596,855) | ~74 % (1,111)  | 96.55 % |
| ~49.7 % (446,000) | ~66 % (990)    | 96.33 % |
| ~9.9 % (89,200)   | ~17 % (255)    | 82.10 % |
| ~0.56 % (5,003)   | ~2 % (26)      | 22.81 % |

### Clustering method

- **Algorithm:** `AgglomerativeClustering` (scikit-learn) with **average linkage** and precomputed cosine distance matrix.
- **Distance threshold range:** searched over `[0.4, 1.0]` (11 values per layer, 6 layers → full grid of 1,771,561 combinations in `full_search.py`).

### Distance thresholds used (symmetry vs. time analysis)

Three threshold configurations were evaluated during training:

| Configuration | fib thresholds (L1, L2, L3) | opfib thresholds (L1, L2, L3) |
|---------------|-----------------------------|-------------------------------|
| A | 0.75, 0.65, 0.15 | 1.0, 1.0, 1.0 |
| B | 0.80, 0.80, 0.80 | 1.0, 1.0, 1.0 |
| C | 1.50, 1.50, 1.50 | 0.8, 0.8, 0.3 |

## Limitations

- Architecture and hyperparameters are tuned for a controlled MNIST symmetry study, not for benchmark-competitive performance.
- Symmetry detection depends on clustering distance thresholds set per layer — results may be sensitive to this choice.
- `ablation_version` uses random node selection (`randperm`) with no fixed seed; results should be averaged over multiple runs for a fair baseline comparison.
- Findings on MNIST/MLP may not generalize directly to convolutional or larger-scale architectures without further validation.

## Ethical Considerations

- Standard MNIST digit classification; no personally identifiable or sensitive data involved.
- No foreseeable direct societal risk; this is foundational research into network structure and compression methods.

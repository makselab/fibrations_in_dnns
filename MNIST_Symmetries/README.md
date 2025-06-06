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
  - Requires `epsilon` parameter (100 values between `0` and `30`).
  - *Note: `epsilon` depends on activity distribution (normalization WIP).*
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
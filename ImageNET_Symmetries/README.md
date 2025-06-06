# Training Convolutional Network using SGD, Continual Learning and Breaking Symmetry

## Initial Setup

Define the correct paths in the .sh scripts for saving the data (`PATHDATA`) and the results (`PATHRES`).

## Script Descriptions

### Core Functionality
- **`main_imagenet.sh'**: The network training runs for 5000 tasks using a configuration file. Each configuration file is associated with a specific training method.
  - 01 → exp_01 → Regular SGD
  - 02 → exp_02 → Continual Learning
  - 03 → exp_03 → Breaking Symmetry

- **`coloring_imagenet.py`**: Compute the symmetries (fibrations, opfibrations, coverings) for each experiment.
- **`plotting_imagenet.py`**:  For each experiment, plot the following as a function of training time:
  - Number of symmetries
  - Size of the fibers/opfibers/covers
  - Transition matrices
  - Performance metrics

---

### Results Folder Structure
- ImageNet/  
  - exp_01/  
    - run_1/
      -  plots/
      -  symmetries/
          - fibrations/
          - covering/
          - opfibrations/
    - run_N/
      - ...
  - exp_02/  
    - ... 

 ---

### Reproducing the results
To reproduce the results shown in the `results/` directory, execute the following scripts in order:
- main_imagenet.sh exp
- coloring_imagenet.sh exp
- plotting_imagenet.sh exp

where exp = 01, 02, 03

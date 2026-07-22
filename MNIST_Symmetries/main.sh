#!/bin/bash

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
PATHdata='./'
PATHcfgs='./cfgfiles/'
PATHcoloring='./results/'
exp_name='exp_'$1
epoch_idx=599

# ===========================================================
# Main Training
python3 src/training.py -exp_name $exp_name\
						-PATHtrain $PATHtrain\
						-PATHcfg $PATHcfgs

# ===========================================================
# Collapse and Ablation Model
python3 src/compression.py -exp $exp_name\
						-PATHtrain $PATHtrain\
						-PATHresults $PATHcoloring\
						-epoch $epoch_idx

# ===========================================================
# Evaluation Compressed
python3 src/evaluation_compressed.py -exp $exp_name\
 						-PATHtrain $PATHtrain\
 						-PATHresults $PATHcoloring\
 						-PATHdata $PATHdata\
 						-epoch $epoch_idx

# ===========================================================
# Symmetry vs Time (Fibers). Threshold depends on layer.
python3 src/symmetry_vs_time.py -exp_name $exp_name\
								-PATHtrain $PATHtrain\
								-PATHresults $PATHcoloring\
								-distance_thrs 0.75 0.65 0.15 1.0 1.0 1.0

# ===========================================================
# Symmetry vs Time (Fbers). Same thresolds for all layers (0.8).
python3 src/symmetry_vs_time.py -exp_name $exp_name\
								-PATHtrain $PATHtrain\
								-PATHresults $PATHcoloring\
								-distance_thrs 0.8 0.8 0.8 1.0 1.0 1.0

# ===========================================================
# Symmetry vs Time (Opfibers).
python3 src/symmetry_vs_time.py -exp_name $exp_name\
								-PATHtrain $PATHtrain\
								-PATHresults $PATHcoloring\
								-distance_thrs 1.5 1.5 1.5 0.8 0.8 0.3
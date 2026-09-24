#!/bin/bash

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
PATHdata='./'
PATHresults='./results/'
exp_name='exp_'$1
version=${2:-v2}
s_reg=${3:-1e-3}
epoch_idx=599

export PYTHONPATH=src

# Distance threshold values to sweep (20 points in (0, 0.05])
distance_threshold_values=$(python3 -c "import numpy as np; print(' '.join(f'{x:.4f}' for x in np.linspace(0, 0.1, 501)[1:]))")

# ===========================================================
# Loss Coloring & Collapse

echo 'Loss Compression'
python3 src/compression_loss.py -exp_name $exp_name\
                                -PATHtrain $PATHtrain\
                                -PATHresults $PATHresults\
                                -PATHdata $PATHdata\
                                -epoch $epoch_idx\
                                -distance_threshold $distance_threshold_values\
                                -version $version\
                                -s_reg $s_reg

# ===========================================================
# Evaluation Loss Models

echo 'Evaluation Loss Models'
python3 src/evaluation_loss.py -exp_name $exp_name\
                               -PATHtrain $PATHtrain\
                               -PATHresults $PATHresults\
                               -PATHdata $PATHdata\
                               -epoch $epoch_idx\
                               -distance_threshold $distance_threshold_values\
                               -version $version

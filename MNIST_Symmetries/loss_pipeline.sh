#!/bin/bash

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
PATHdata='./'
PATHresults='./results/'
exp_name='exp_'$1
epoch_idx=599

export PYTHONPATH=src

# F_max budget values to sweep
F_max_values="0.1 0.5 1.0 2.0 5.0"

# ===========================================================
# Loss Coloring & Collapse

echo 'Loss Compression'
python3 src/compression_loss.py -exp_name $exp_name\
                                -PATHtrain $PATHtrain\
                                -PATHresults $PATHresults\
                                -PATHdata $PATHdata\
                                -epoch $epoch_idx\
                                -F_max $F_max_values

# ===========================================================
# Evaluation Loss Models

echo 'Evaluation Loss Models'
python3 src/evaluation_loss.py -exp_name $exp_name\
                               -PATHtrain $PATHtrain\
                               -PATHresults $PATHresults\
                               -PATHdata $PATHdata\
                               -epoch $epoch_idx\
                               -F_max $F_max_values

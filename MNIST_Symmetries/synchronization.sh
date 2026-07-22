#!/bin/bash

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
PATHdata='./'
PATHresults='./results/'
exp_name='exp_'$1
epoch_idx=599

export PYTHONPATH=src

# ===========================================================
# Generate Activity
python3 src/synchronization/generate_activity.py -exp_name $exp_name\
                                                  -PATHtrain $PATHtrain\
                                                  -PATHresults $PATHresults\
                                                  -PATHdata $PATHdata\
                                                  -epoch $epoch_idx

# ===========================================================
# Clustering
python3 src/synchronization/clusters.py -exp_name $exp_name\
                                         -PATHresults $PATHresults\
                                         -epoch $epoch_idx

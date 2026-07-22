#!/bin/bash

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
PATHdata='./'
PATHresults='./results/'
exp_name='exp_'$1
epoch_idx=599
num_thrs=100
max_thr=1.0

export PYTHONPATH=src

# ===========================================================
# Generate Activity
python3 src/synchronization/generate_activity.py -exp_name $exp_name\
                                                  -PATHtrain $PATHtrain\
                                                  -PATHresults $PATHresults\
                                                  -PATHdata $PATHdata\
                                                  -epoch $epoch_idx

# ===========================================================
# Generate Activity - Mean Class
python3 src/synchronization/generate_activity_mean_class.py -exp_name $exp_name\
                                                             -PATHtrain $PATHtrain\
                                                             -PATHresults $PATHresults\
                                                             -PATHdata $PATHdata\
                                                             -epoch $epoch_idx

# ===========================================================
# Clustering
python3 src/synchronization/clusters.py -exp_name $exp_name\
                                         -PATHresults $PATHresults\
                                         -epoch $epoch_idx

# ===========================================================
# Clustering per Class
python3 src/synchronization/clusters_per_class.py -exp_name $exp_name\
                                                   -PATHresults $PATHresults\
                                                   -epoch $epoch_idx

# ===========================================================
# Generate Fibration Coloring
python3 src/synchronization/generate_fibration_coloring.py -exp_name $exp_name\
                                                            -PATHtrain $PATHtrain\
                                                            -PATHresults $PATHresults\
                                                            -epoch $epoch_idx\
                                                            -num_thrs $num_thrs\
                                                            -max_thr $max_thr

# ===========================================================
# Matching
python3 src/synchronization/matching.py -exp_name $exp_name\
                                         -PATHresults $PATHresults\
                                         -epoch $epoch_idx

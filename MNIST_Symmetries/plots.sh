#!/bin/bash

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
PATHresults='./results/'
exp_name='exp_'$1

# ===========================================================
# Metrics during training
python3 src/plots/plt_metrics_training.py -exp_name $exp_name\
                                           -PATHresults $PATHresults\
                                           -PATHtrain $PATHtrain

# ===========================================================
# Symmetry vs time
python3 src/plots/plt_symmetry_vs_time.py -exp_name $exp_name\
                                           -PATHresults $PATHresults\
                                           -PATHtrain $PATHtrain

# ===========================================================
# Optimal curves
python3 src/plots/plt_optimal_curve.py -exp_name $exp_name\
                                        -PATHresults $PATHresults

# ===========================================================
# Pareto frontier
python3 src/plots/plt_pareto_frontier.py -exp_name $exp_name\
                                          -PATHresults $PATHresults

# ===========================================================
# Pareto accuracy
python3 src/plots/plt_pareto_acurracy.py -exp_name $exp_name\
                                          -PATHresults $PATHresults

# ===========================================================
# Compressed accuracy
python3 src/plots/plt_compressed_acurracy.py -exp_name $exp_name\
                                              -PATHresults $PATHresults

# ===========================================================
# dL approximate vs real
python3 src/plots/plt_dL.py -exp_name $exp_name\
                                        -PATHresults $PATHresults

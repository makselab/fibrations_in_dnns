#!/bin/bash

# ===========================================================
# Initial Configuration - Parameters
PATHtrain='./train_dir/'
PATHres='./results/'
exp_name='exp_'$1

# # ===========================================================
# # Main Search

# for opf_thr in $(seq 1 -0.05 +0.45); do
#     echo 'Running Opf Thr: '$opf_thr

# 	python3 src/search.py -exp_name $exp_name\
# 							-PATHtrain $PATHtrain\
# 							-PATHres $PATHres\
# 							-opfiber_threshold $opf_thr

#     echo '--------------------------'
# done

# ===========================================================
# Main Search

python3 src/full_search.py -exp_name $exp_name\
						-PATHtrain $PATHtrain\
						-PATHres $PATHres
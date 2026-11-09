#!/bin/bash

# ===========================================================
# Initial Configuration
PATHtrain='./train_dir/'
PATHdata='./'
PATHcfgs='./cfgfiles/'
PATHresults='./results/'
cfg_name='exp_01'
epoch_idx=599
N_RUNS=${1:-10}
THR_MIN=0.0
THR_MAX=1.0
N_VALS=20
OPFIB_THR=1.0

export PYTHONPATH=src

# ===========================================================
for i in $(seq 1 $N_RUNS); do
    exp_name=$(printf 'exp_01_r%02d' $i)
    echo "======================================================="
    echo "=== Run $i / $N_RUNS : $exp_name ==="
    echo "======================================================="

    # Create results directory for this run
    mkdir -p ${PATHresults}${exp_name}

    # Training
    python3 src/training.py \
        -exp_name  $exp_name \
        -PATHtrain $PATHtrain \
        -PATHcfg   $PATHcfgs \
        -cfg_name  $cfg_name \
        -seed      $i

    # Save training curves (accuracy, loss) as CSV before cleanup
    python3 -c "
import torch, pandas as pd
exp_dir = '${PATHtrain}${exp_name}'
acc = torch.load(exp_dir + '/accuracy.pth').tolist()
lt  = torch.load(exp_dir + '/loss_train.pth')
lo  = torch.load(exp_dir + '/loss_test.pth')
lt  = [x.item() if hasattr(x, 'item') else float(x) for x in lt]
lo  = [x.item() if hasattr(x, 'item') else float(x) for x in lo]
pd.DataFrame({'step': range(len(acc)), 'accuracy': acc, 'loss_train': lt, 'loss_test': lo}).to_csv('${PATHresults}${exp_name}/training_curves.csv', index=False)
print('Saved training_curves.csv')
"

    # Symmetry vs time (fibration covering)
    python3 src/symmetry_vs_time.py \
        -exp_name     $exp_name \
        -PATHtrain    $PATHtrain \
        -PATHresults  $PATHresults \
        -distance_thrs 0.75 0.65 0.15 1.0 1.0 1.0

    # Delete intermediate checkpoints, keep only the last one
    find ${PATHtrain}${exp_name}/checkpoints/ -name 'model_batch_*.pth' ! -name "model_batch_${epoch_idx}.pth" -delete

    # Copy optimal curves from exp_01 (same architecture)
    mkdir -p ${PATHresults}${exp_name}/optimal_curves
    cp ${PATHresults}exp_01/optimal_curves/*.csv ${PATHresults}${exp_name}/optimal_curves/

    # Compression (fibration + ablation + pruning + PFP + EigenDamage)
    python3 src/compression.py \
        -exp_name    $exp_name \
        -PATHtrain   $PATHtrain \
        -PATHresults $PATHresults \
        -PATHdata    $PATHdata \
        -epoch       $epoch_idx

    # Evaluation compressed models
    python3 src/evaluation_compressed.py \
        -exp_name    $exp_name \
        -PATHtrain   $PATHtrain \
        -PATHresults $PATHresults \
        -PATHdata    $PATHdata \
        -epoch       $epoch_idx

    # Delete compressed model weights (model_batch_0.pth from all compressed subdirs)
    find ${PATHtrain} -maxdepth 3 -name 'model_batch_0.pth' -path "*${exp_name}*" -delete

    # Fibration threshold sweep (no models saved to disk)
    #python3 src/sensibility_analysis.py \
    #    -exp_name    $exp_name \
    #    -PATHtrain   $PATHtrain \
    #    -PATHresults $PATHresults \
    #    -PATHdata    $PATHdata \
    #    -epoch       $epoch_idx \
    #    -thr_min     $THR_MIN \
    #    -thr_max     $THR_MAX \
    #    -n_vals      $N_VALS \
    #    -opfib_thr   $OPFIB_THR

    # Delete all train_dir content for this run (models + dirs)
    rm -rf ${PATHtrain}${exp_name}
    find ${PATHtrain} -maxdepth 1 -type d -name "${exp_name}_epoch_*" -exec rm -rf {} \;

done

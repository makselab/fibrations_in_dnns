#!/bin/bash

_PATH='./'
TITLE='exp_'$1
NUM_RUNS=1
NUM_TASKS=5000
PATHDATA='/media/osvaldo/Seagate Basic/ImageNet/'
PATHRES='/media/osvaldo/OMV5TB/BreakingSymmetry/results/ImageNet/'$TITLE'/'

export PYTHONPATH=$(pwd):$PYTHONPATH

# -------------------------------------------------------------

for (( IDX_RUN=1; IDX_RUN<=$NUM_RUNS; IDX_RUN++ )) do
    echo 'Training Stage - Model' $TITLE ' - Run - ' $IDX_RUN

    python3.8 src/main_imagenet.py \
        --cfgfilename $_PATH'cfgfiles/imagenet/'$TITLE'.json'\
        --num_tasks $NUM_TASKS\
        --idx_run $IDX_RUN\
        --datapath "$PATHDATA"\
        --respath $PATHRES'run_'$IDX_RUN'/'

done

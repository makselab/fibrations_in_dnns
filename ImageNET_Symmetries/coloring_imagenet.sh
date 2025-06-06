#!/bin/bash

TITLE='exp_'$1
NUM_TASKS=$2
PATHDATA='/media/osvaldo/OMV5TB/BreakingSymmetry/results/ImageNet/'$TITLE'/weights/'
PATHRES='/home/osvaldo/Documents/CCNY/Project_BreakingSymmetry/results/ImageNet/'$TITLE'/'

export PYTHONPATH=$(pwd):$PYTHONPATH

# -------------------------------------------------------------
echo 'Coloring - Fibrations - ' $TITLE

python3.8 src/coloring/ImageNet/fibrations.py \
	--dataPATH "$PATHDATA"\
	--resPATH "$PATHRES"\
    --num_tasks $NUM_TASKS
	
# -------------------------------------------------------------
echo 'Coloring - Opfibrations - ' $TITLE

python3.8 src/coloring/ImageNet/optfibrations.py \
	--dataPATH "$PATHDATA"\
	--resPATH "$PATHRES"\
    --num_tasks $NUM_TASKS
	
# -------------------------------------------------------------
echo 'Coloring - Covering - ' $TITLE

python3.8 src/coloring/ImageNet/covering.py \
	--resPATH "$PATHRES"

# -------------------------------------------------------------

#!/bin/bash

TITLE='exp_'$1
NUM_TASKS=$2
PATHDATA='/media/osvaldo/OMV5TB/BreakingSymmetry/results/ImageNet/'$TITLE'/run_1/performance/'
PATHRES='/home/osvaldo/Documents/CCNY/Project_BreakingSymmetry/results/ImageNet/'$TITLE'/'
SYMMETRIES=("fibrations" "optfibrations" "covering")
PLOT_TASKS=(0 1 $((NUM_TASKS - 100)) $((NUM_TASKS - 99)))
PLOT_MTX=(0 $((NUM_TASKS - 100)))

export PYTHONPATH=$(pwd):$PYTHONPATH

# -------------------------------------------------------------

for SYMMETRY in "${SYMMETRIES[@]}"; do
	echo 'Plot Num Fibers - vs -  Time - ' $SYMMETRY $TITLE

	python3.8 src/plots/plt_num_fibers_vs_time.py \
		--dataPATH "$PATHRES"\
		--symmetry $SYMMETRY\
		--layer_size 32 64 512 128 128\
		--num_epochs 6\
		--window_size 20

	echo 'Plot Sizes Fibers - vs -  Time - ' $SYMMETRY $TITLE

	for TASK_IDX in "${PLOT_TASKS[@]}"; do
		python3.8 src/plots/plt_sizes_vs_time.py \
			--dataPATH "$PATHRES"\
			--symmetry $SYMMETRY\
			--task_idx $TASK_IDX
	done 

	echo 'Plot Transitions - ' $SYMMETRY $TITLE

	for TASK_IDX in "${PLOT_MTX[@]}"; do
		python3.8 src/plots/plt_transitions_symmetries.py \
			--dataPATH "$PATHRES"\
			--symmetry $SYMMETRY\
			--task_idx $TASK_IDX
	done 
done

# -------------------------------------------------------------
echo 'Plot Performance - ' $TITLE

python3.8 src/plots/plt_performance.py \
	--dataPATH "$PATHDATA"\
	--resPATH "$PATHRES"\
    --num_tasks $NUM_TASKS\
    --ymin 0.8\
    --window_size 100

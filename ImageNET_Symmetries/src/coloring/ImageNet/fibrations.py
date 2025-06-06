# =================================================================
# MODULES.

from torch import load
from pandas import DataFrame
import argparse
from src.coloring.coloring import fibration_conv2d, fibration_linear

# =================================================================
# PARAMETERS - VARIABLES.

parser = argparse.ArgumentParser()
parser.add_argument('--dataPATH', type=str)
parser.add_argument('--resPATH', type=str)
parser.add_argument('--num_tasks', type=int) # 5000 
args = parser.parse_args()

epochs  = [0,50,100,150,200,250]
threshold = 0.8 

# 0,2,4 : Conv2d
# 6,8,10: Linear

# =================================================================
# PROCESSING.

fibers_vs_time = [[],[],[],[],[]] 

for task_idx in range(args.num_tasks):
	for ep_idx in epochs:
		print('Epoch - ' + str(ep_idx) + ' Task - ' + str(task_idx))

		weights = load(args.dataPATH + 'task_idx_' + str(task_idx) + '_epoch_' + str(ep_idx) + '.pth')

		# First Layer ---------------------------------------
		clusters = fibration_conv2d(weights=weights['layers.0.weight'], 
						in_clusters=None, 
						threshold=threshold, 
						first_layer = True,
						pooling=None)

		fibers_vs_time[0].append(clusters)

		# Second Layer ---------------------------------------
		clusters = fibration_conv2d(weights=weights['layers.2.weight'], 
									in_clusters=clusters, 
									threshold=threshold, 
									first_layer = False,
									pooling=None)

		fibers_vs_time[1].append(clusters)

		# Third Layer ---------------------------------------

		clusters = fibration_conv2d(weights=weights['layers.4.weight'], 
									in_clusters=clusters, 
									threshold=threshold, 
									first_layer = False,
									pooling=4)

		fibers_vs_time[2].append(clusters)

		# 4th Layer ---------------------------------------
		
		clusters = fibration_linear(weights=weights['layers.6.weight'], 
									in_clusters=clusters, 
									threshold=threshold, 
									first_layer = False)

		fibers_vs_time[3].append(clusters)

		# 5th Layer ---------------------------------------

		clusters = fibration_linear(weights=weights['layers.8.weight'], 
									in_clusters=clusters, 
									threshold=threshold, 
									first_layer = False)

		fibers_vs_time[4].append(clusters)

for idx_LL in range(5):
	df = DataFrame({f'T_{i+1}': array for i, array in enumerate(fibers_vs_time[idx_LL])})
	df.to_csv(args.resPATH + 'symmetries/fibrations/Layer_' + str(idx_LL) + '.csv', index = False, header= False)

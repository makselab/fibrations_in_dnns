# =================================================================
# MODULES.

from torch import load
from pandas import DataFrame
import argparse
from src.coloring.coloring import opfibration_conv2d, opfibration_linear

# =================================================================
# PARAMETERS - VARIABLES.

parser = argparse.ArgumentParser()
parser.add_argument('--dataPATH', type=str)
parser.add_argument('--resPATH', type=str)
parser.add_argument('--num_tasks', type=int) # 5000 
args = parser.parse_args()

epochs  = [0,50,100,150,200,250]
threshold = 0.5

# 0,2,4 : Conv2d
# 6,8,10: Linear

# =================================================================
# PROCESSING.

fibers_vs_time = [[],[],[],[],[]] 

for task_idx in range(args.num_tasks):
	for ep_idx in epochs:
		print('Epoch - ' + str(ep_idx) + ' Task - ' + str(task_idx))

		weights = load(args.dataPATH + 'task_idx_' + str(task_idx) + '_epoch_' + str(ep_idx) + '.pth')

		# 5th Layer ---------------------------------------
		clusters = opfibration_linear(weights=weights['layers.10.weight'], 
							out_clusters=None, 
							threshold=threshold, 
							last_layer = True)

		fibers_vs_time[4].append(clusters)

		# 4th Layer ---------------------------------------
		clusters = opfibration_linear(weights=weights['layers.8.weight'], 
							out_clusters=clusters, 
							threshold=threshold, 
							last_layer = False)

		fibers_vs_time[3].append(clusters)

		# Third Layer ---------------------------------------
		clusters = opfibration_linear(weights=weights['layers.6.weight'], 
							out_clusters=clusters, 
							threshold=threshold, 
							last_layer = False)

		fibers_vs_time[2].append(clusters)

		# # Transformation -------------------------------------
		# clusters = clusters[::4]

		# Second Layer ---------------------------------------
		clusters = opfibration_conv2d(weights = weights['layers.4.weight'], 
							out_clusters = clusters, 
							threshold = threshold, 
							last_layer = False,
							pooling = 4)

		fibers_vs_time[1].append(clusters)

		# First Layer ---------------------------------------
		clusters = opfibration_conv2d(weights = weights['layers.2.weight'], 
							out_clusters = clusters, 
							threshold = threshold, 
							last_layer = False,
							pooling = None)

		fibers_vs_time[0].append(clusters)

for idx_LL in range(5):
	df = DataFrame({f'T_{i+1}': array for i, array in enumerate(fibers_vs_time[idx_LL])})
	df.to_csv(args.resPATH + 'symmetries/optfibrations/Layer_' + str(idx_LL) + '.csv', index = False, header= False)
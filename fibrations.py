# =================================================================
# MODULES.

from torch import load
import pandas as pd
from coloring import fibration_conv2d, fibration_linear

# =================================================================
# PARAMETERS - VARIABLES.

epochs = [22000]
threshold = 0.7

# =================================================================
# PROCESSING.

fibers_vs_time = [[],[],[],[]] 

for ep_idx in epochs:

	weights = load('/home/osvaldo/Documents/STEM-AI/Project_PufferAI/checkpoints/policy_epoch_' + str(ep_idx) + '.pt')

	clusters = fibration_conv2d(weights = weights['network.0.weight'], 
								  	in_clusters = None, 
								  	threshold =threshold, 
								  	first_layer = True)

	fibers_vs_time[0].append(clusters)

	clusters = fibration_conv2d(weights = weights['network.2.weight'], 
								  	in_clusters = clusters, 
								  	threshold =threshold, 
								  	first_layer = False)

	fibers_vs_time[1].append(clusters)

	clusters = fibration_conv2d(weights = weights['network.4.weight'], 
								  	in_clusters = clusters, 
								  	threshold =threshold, 
								  	first_layer = False)

	fibers_vs_time[2].append(clusters)

	clusters = [x for x in clusters for _ in range(54)]

	clusters = fibration_linear(weights = weights['network.7.weight'], 
								  	in_clusters = clusters, 
								  	threshold =threshold, 
								  	first_layer = False)

	fibers_vs_time[3].append(clusters)


for idx_LL in range(4):
	df = pd.DataFrame({f'T_{i+1}': array for i, array in enumerate(fibers_vs_time[idx_LL])})
	df.to_csv('/home/osvaldo/Documents/STEM-AI/Project_PufferAI/results/symmetries/fibrations/Layer_' + str(idx_LL) + '.csv', index = False, header= False)
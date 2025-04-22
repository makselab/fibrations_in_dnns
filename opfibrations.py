# =================================================================
# MODULES.

from torch import load, cat, tensor, mm
import pandas as pd
from coloring import opfibration_conv2d, opfibration_linear
from sklearn.cluster import AgglomerativeClustering

# =================================================================
# PARAMETERS - VARIABLES.

epochs = [22000]
threshold = 0.7

# =================================================================
# PROCESSING.

fibers_vs_time = [[],[],[],[]] 

for ep_idx in epochs:

	weights = load('/home/osvaldo/Documents/STEM-AI/Project_PufferAI/checkpoints/policy_epoch_' + str(ep_idx) + '.pt')

	w_actor = weights['actor.weight']
	w_critc = weights['critic.weight']

	clusters = opfibration_linear(weights=cat((w_actor,w_critc), dim=0), 
								out_clusters=None, 
								threshold=threshold, 
								last_layer = True)

	fibers_vs_time[3].append(clusters)

	# -----------------------------------------------------

	clusters = opfibration_linear(weights = weights['network.7.weight'], 
								out_clusters=clusters, 
								threshold=threshold, 
								last_layer = False)

	# Flatten OptFibration
	clusters_matrix = tensor(clusters)
	clusters_matrix = clusters_matrix.view(-1, 54)

	distance = 1 - mm(clusters_matrix,clusters_matrix.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = clustering.fit_predict(distance)

	fibers_vs_time[2].append(clusters)

	# -----------------------------------------------------

	clusters = opfibration_conv2d(weights = weights['network.4.weight'], 
								out_clusters = clusters, 
								threshold =threshold, 
								last_layer = False)

	fibers_vs_time[1].append(clusters)

	# -----------------------------------------------------

	clusters = opfibration_conv2d(weights = weights['network.2.weight'], 
							  	out_clusters = clusters, 
							  	threshold =threshold, 
							  	last_layer = False)

	fibers_vs_time[0].append(clusters)

	# -----------------------------------------------------

for idx_LL in range(4):
	df = pd.DataFrame({f'T_{i+1}': array for i, array in enumerate(fibers_vs_time[idx_LL])})
	df.to_csv('/home/osvaldo/Documents/STEM-AI/Project_PufferAI/results/symmetries/opfibrations/Layer_' + str(idx_LL) + '.csv', index = False, header= False)




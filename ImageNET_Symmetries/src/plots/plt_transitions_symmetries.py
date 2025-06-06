# =================================================================
# MODULES

import matplotlib.pyplot as plt
from pandas import read_csv, crosstab
from numpy import zeros
from seaborn import heatmap
import argparse

# =================================================================
# PARAMETERS - VARIABLES.

parser = argparse.ArgumentParser()
parser.add_argument('--dataPATH', type=str)
parser.add_argument('--symmetry', type=str)
parser.add_argument('--task_idx', type=int)
args = parser.parse_args()

len_task = 2
num_epochs = 6
annot = False
cbar=True

# ---------------------------------------------------------------------
def calcular_matriz_transiciones(df_clusters):
    num_tiempos = df_clusters.shape[1]
    clusters_unicos = sorted(set(df_clusters.values.flatten()))
    num_clusters = len(clusters_unicos)
    
    cluster_a_indice = {cluster: i for i, cluster in enumerate(clusters_unicos)}
    transiciones = zeros((num_tiempos - 1, num_clusters, num_clusters), dtype=int)
    
    for t in range(num_tiempos - 1):
        for n in range(df_clusters.shape[0]):
            cluster_actual = df_clusters.iloc[n, t]
            cluster_siguiente = df_clusters.iloc[n, t + 1]
            transiciones[t, cluster_a_indice[cluster_actual], cluster_a_indice[cluster_siguiente]] += 1
    
    return transiciones, clusters_unicos
# ---------------------------------------------------------------------

for idx_LL in range(0,5):
	df = read_csv(args.dataPATH + 'symmetries/' + args.symmetry + '/Layer_' + str(idx_LL) + '.csv', header=None)

	for t in range(args.task_idx*num_epochs, (args.task_idx+len_task)*num_epochs):

		data = df.iloc[:,t:t+2]

		transitions = crosstab(data[t], data[t+1])
		transitions = transitions.div(transitions.sum(axis=1), axis=0)

		fig, axs = plt.subplots(figsize=(8, 8))
		heatmap(transitions, annot=annot, fmt="d", cmap="YlGnBu", cbar=cbar, ax=axs)
		axs.set_xlabel("Next step")
		axs.set_ylabel("Actual step")
	    
		actual_idx_epoch = t % num_epochs 
		actual_idx_task  = t // num_epochs
		next_idx_epoch   = (t+1) % num_epochs 
		next_idx_task    = (t+1) // num_epochs

		axs.set_title(f"Task {actual_idx_task} ({actual_idx_epoch*50}) to Task {next_idx_task} ({next_idx_epoch*50})")

		fig.savefig(args.dataPATH + 'symmetries/' + args.symmetry + '/transitions/Layer_' + str(idx_LL)+ '_Task_' + str(actual_idx_task) + '_Epoch_' + str(actual_idx_epoch) + '.svg', format='svg')

		plt.close(fig)

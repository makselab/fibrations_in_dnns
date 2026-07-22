# =====================================================
# MODULES

import argparse

import torch
import numpy as np
from sklearn.metrics import pairwise_distances

from symmetries.clustering import make_algorithm

# =====================================================
# Load args, paths.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-epoch', type=int, required=True, help='Epoch')
parser.add_argument('-num_distance_thr', type=int, default=100, help='Number of epsilon thresholds')
parser.add_argument('-max_epsilon', type=float, default=2.0, help='Max distance threshold (epsilon)')

args = parser.parse_args()

clustering_method = {'name': 'agg_clustering', 'cfg': {'linkage': 'complete'}}

# =====================================================
# Load activity (random inputs)

results_folder = args.PATHresults + args.exp_name + '/synchronization/'
activities = torch.load(results_folder + 'activity_random_input_batch_' + str(args.epoch) + '.pth')

epsilons = torch.linspace(0, args.max_epsilon, args.num_distance_thr)
num_eps = len(epsilons)
num_layers = len(activities)

# =====================================================
# CLUSTERING

num_clusters_layers = torch.zeros(num_eps, num_layers)
clusters_layers = {f'L{idx_layer+1}': np.zeros((num_eps, activities[idx_layer].shape[1])) for idx_layer in range(num_layers)}
clusters_layers['eps'] = epsilons

for idx_layer, h in enumerate(activities):
	h_normalized = (h - h.mean(dim=0)) / h.std(dim=0)
	h_normalized = h_normalized.cpu()
	h_normalized = h_normalized / torch.norm(h_normalized, p=2, dim=0, keepdim=True)

	distance_matrix = pairwise_distances(h_normalized.T, metric='euclidean')

	for idx_eps, eps in enumerate(epsilons):
		algorithm = make_algorithm(clustering_method['name'], clustering_method['cfg'], eps.item())
		clusters = algorithm(distance_matrix)

		num_clusters_layers[idx_eps, idx_layer] = len(np.unique(clusters))
		clusters_layers[f'L{idx_layer+1}'][idx_eps, :] = clusters

torch.save(clusters_layers, results_folder + 'clusters_batch_' + str(args.epoch) + '.pth')
torch.save(num_clusters_layers, results_folder + 'num_clusters_batch_' + str(args.epoch) + '.pth')
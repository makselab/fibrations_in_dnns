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

results_folder = args.PATHresults + args.exp_name + '/synchronization/'

# =====================================================
# Load activity

data = torch.load(results_folder + 'activity_batch_' + str(args.epoch) + '.pth')
mean_activities = torch.load(results_folder + 'activity_mean_class_batch_' + str(args.epoch) + '.pth')

epsilons = np.linspace(0, args.max_epsilon, args.num_distance_thr)
num_eps = len(epsilons)
num_layers = len(data['activity'])
num_classes = len(torch.unique(data['labels']))

idxs_samples = [data['labels'] == cl for cl in range(num_classes)]

# =====================================================
# CLUSTERING PER CLASS

for cl in range(num_classes):

    clusters_layers = {f'L{idx_l+1}': np.zeros((num_eps, data['activity'][idx_l].shape[1])) for idx_l in range(num_layers)}
    clusters_layers['eps'] = epsilons
    n_clusters_layers = torch.zeros(num_eps, num_layers)

    mean_clusters_layers = {f'L{idx_l+1}': np.zeros((num_eps, data['activity'][idx_l].shape[1])) for idx_l in range(num_layers)}
    mean_clusters_layers['eps'] = epsilons
    mean_n_clusters_layers = torch.zeros(num_eps, num_layers)

    distance_matrices = {}
    distance_matrices_mean = {}

    for idx_l, h in enumerate(data['activity']):
        layer_key = f'L{idx_l + 1}'

        # Subset of class --------------------------------
        H = h[idxs_samples[cl], :]
        h_normalized = (H - H.mean(dim=0)) / H.std(dim=0)
        h_normalized = h_normalized.cpu()
        h_normalized = h_normalized / torch.norm(h_normalized, p=2, dim=0, keepdim=True)
        distance_matrix = pairwise_distances(h_normalized.T, metric='euclidean')
        distance_matrices[layer_key] = distance_matrix

        # Mean of class ----------------------------------
        H_mean = mean_activities[idx_l][[cl], :].cpu()
        H_mean = H_mean / torch.norm(H_mean, p=2, dim=0, keepdim=True)
        distance_matrix_mean = pairwise_distances(H_mean.T, metric='euclidean')
        distance_matrices_mean[layer_key] = distance_matrix_mean

        for idx_eps, eps in enumerate(epsilons):
            algorithm = make_algorithm(clustering_method['name'], clustering_method['cfg'], eps.item())

            clusters = algorithm(distance_matrix)
            n_clusters_layers[idx_eps, idx_l] = len(np.unique(clusters))
            clusters_layers[layer_key][idx_eps, :] = clusters

            mean_clusters = algorithm(distance_matrix_mean)
            mean_n_clusters_layers[idx_eps, idx_l] = len(np.unique(mean_clusters))
            mean_clusters_layers[layer_key][idx_eps, :] = mean_clusters

    torch.save(clusters_layers, results_folder + 'clusters_class_' + str(cl) + '_batch_' + str(args.epoch) + '.pth')
    torch.save(n_clusters_layers, results_folder + 'num_clusters_class_' + str(cl) + '_batch_' + str(args.epoch) + '.pth')
    torch.save(mean_clusters_layers, results_folder + 'mean_clusters_class_' + str(cl) + '_batch_' + str(args.epoch) + '.pth')
    torch.save(mean_n_clusters_layers, results_folder + 'mean_num_clusters_class_' + str(cl) + '_batch_' + str(args.epoch) + '.pth')
    torch.save(distance_matrices, results_folder + 'distance_matrices_class_' + str(cl) + '_batch_' + str(args.epoch) + '.pth')
    torch.save(distance_matrices_mean, results_folder + 'distance_matrices_mean_class_' + str(cl) + '_batch_' + str(args.epoch) + '.pth')

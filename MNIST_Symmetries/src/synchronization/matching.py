import argparse

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

# =====================================================
# Load args, paths.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-epoch', type=int, required=True, help='Epoch')

args = parser.parse_args()

results_folder = args.PATHresults + args.exp_name + '/'

# =============================================================

def one_to_one_cluster_similarity(labels_A, labels_B):
    """
    Compute the best one-to-one cluster alignment similarity score
    between two clusterings using the Hungarian algorithm.

    Parameters:
    - labels_A: list or array of cluster labels for clustering A
    - labels_B: list or array of cluster labels for clustering B

    Returns:
    - normalized_overlap: float, similarity score between 0 and 1
    """
    cont_matrix = contingency_matrix(labels_A, labels_B)
    cost_matrix = -cont_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_overlap = cont_matrix[row_ind, col_ind].sum()
    return total_overlap / len(labels_A)


def refinement_score(P1, P2):
    total = 0
    for label in np.unique(P1):
        labels_P2 = P2[P1 == label]
        if len(np.unique(labels_P2)) == 1:
            total += 1
    return total / len(np.unique(P1))

# =============================================================
# PARTITIONS - LAYER 1

clusters = torch.load(results_folder + 'synchronization/clusters_batch_' + str(args.epoch) + '.pth')
class_clusters = [torch.load(results_folder + 'synchronization/clusters_class_' + str(cl) + '_batch_' + str(args.epoch) + '.pth') for cl in range(10)]
mean_class_clusters = [torch.load(results_folder + 'synchronization/mean_clusters_class_' + str(cl) + '_batch_' + str(args.epoch) + '.pth') for cl in range(10)]
colors = torch.load(results_folder + 'coloring/fibration_L1_batch_' + str(args.epoch) + '.pth')

layer_key = 'L1'

epsilons = clusters['eps']
num_eps = len(epsilons)

thresholds = colors[layer_key][:, 0]
num_thrs = len(thresholds)

clusters_l = clusters[layer_key]
fibers_l = colors[layer_key][:, 1:].astype(int)
hidden_size = clusters_l.shape[1]

num_clusters = np.array([len(np.unique(row)) for row in clusters_l])
num_colors = np.array([len(np.unique(row)) for row in fibers_l])

# Mean class clusters for L1
mtx_mean_clust = np.stack([mean_class_clusters[cl][layer_key][1, :] for cl in range(10)], axis=1)
_, mean_clusters_l = np.unique(mtx_mean_clust, axis=0, return_inverse=True)

# Class clusters vs epsilon for L1
cls_clus_l = np.zeros((num_eps, hidden_size))
for idx_eps in range(num_eps):
    cls_cluster_eps = np.stack([class_clusters[cl][layer_key][idx_eps] for cl in range(10)], axis=1)
    _, cls_clus_l[idx_eps, :] = np.unique(cls_cluster_eps, axis=0, return_inverse=True)

# =============================================================
# SCORES

matching_score_fib_rndclus = np.zeros((num_eps, num_thrs))
ref_score_fib_in_cls_clus = np.zeros((num_eps, num_thrs))
ref_score_fib_in_m_clus = np.zeros(num_thrs)

matching_score_cl_m_clus = np.zeros(num_eps)
ref_score_rndclus_in_cl_clus = np.zeros(num_eps)
ref_score_rndclus_in_m_clus = np.zeros(num_eps)

for idx_thr in range(num_thrs):
    ff = fibers_l[idx_thr]
    ref_score_fib_in_m_clus[idx_thr] = refinement_score(ff, mean_clusters_l)
    for idx_eps in range(num_eps):
        rc = clusters_l[idx_eps]
        cc = cls_clus_l[idx_eps]
        matching_score_fib_rndclus[idx_eps, idx_thr] = one_to_one_cluster_similarity(ff, rc)
        ref_score_fib_in_cls_clus[idx_eps, idx_thr] = refinement_score(ff, cc)

for idx_eps in range(num_eps):
    rc = clusters_l[idx_eps]
    cc = cls_clus_l[idx_eps]
    ref_score_rndclus_in_cl_clus[idx_eps] = refinement_score(rc, cc)
    ref_score_rndclus_in_m_clus[idx_eps] = refinement_score(rc, mean_clusters_l)
    matching_score_cl_m_clus[idx_eps] = one_to_one_cluster_similarity(cc, mean_clusters_l)

idx_optimal_eps = np.argmax(matching_score_fib_rndclus, axis=0)
optimal_eps = epsilons[idx_optimal_eps]
optimal_num_clusters = num_clusters[idx_optimal_eps]
optimal_match = np.max(matching_score_fib_rndclus, axis=0)

# =============================================================
# Figure 1: optimal matching curves (x = threshold)

fig, axs = plt.subplots(3, 1, figsize=(4, 15))
axs1b = axs[1].twinx()

for ax in axs:
    ax.set_xlabel('Fibration Threshold')

axs[0].set_ylabel('Synchronization Epsilon')
axs[1].set_ylabel('Num Sync Clusters', color='blue')
axs1b.set_ylabel('Num Fibration Colors', color='red')
axs[2].set_ylabel('Matching Score (Fibers vs Sync Clusters)')

axs[0].plot(thresholds, optimal_eps)
axs[1].plot(thresholds, optimal_num_clusters, color='blue')
axs1b.plot(thresholds, num_colors, color='red')
axs[2].plot(thresholds, optimal_match)

fig.savefig(results_folder + 'plots/matching_L1.svg', format='svg')
plt.close(fig)

# =============================================================
# Figure 2: refinement and matching vs epsilon / threshold

fig, axs = plt.subplots(2, 1, figsize=(4, 10))

axs[0].plot(thresholds, ref_score_fib_in_m_clus)
axs[0].set_xlabel('Fibration Threshold')
axs[0].set_ylabel('Refinement: Fibers in Mean Class Clusters')

axs[1].plot(epsilons, ref_score_rndclus_in_cl_clus, label='Sync Clusters < Class Clusters')
axs[1].plot(epsilons, ref_score_rndclus_in_m_clus, label='Sync Clusters < Mean Class Clusters')
axs[1].plot(epsilons, matching_score_cl_m_clus, label='Class Clusters = Mean Class Clusters')
axs[1].set_xlabel('Synchronization Epsilon')
axs[1].legend()

fig.savefig(results_folder + 'plots/matching_refinement_p1_L1.svg', format='svg')
plt.close(fig)

# =============================================================
# Figure 3: heatmaps (eps x thr)

fig, axs = plt.subplots(2, 1, figsize=(8, 12))

im0 = axs[0].pcolormesh(thresholds, epsilons, matching_score_fib_rndclus, cmap='viridis')
im1 = axs[1].pcolormesh(thresholds, epsilons, ref_score_fib_in_cls_clus, cmap='viridis')

fig.colorbar(im0, ax=axs[0])
fig.colorbar(im1, ax=axs[1])

axs[0].set_title('Matching: Fibers vs Sync Clusters')
axs[1].set_title('Refinement: Fibers in Class Clusters')
axs[0].set_ylabel('Synchronization Epsilon')
axs[1].set_ylabel('Synchronization Epsilon')
axs[1].set_xlabel('Fibration Threshold')

fig.savefig(results_folder + 'plots/matching_refinement_p2_L1.svg', format='svg')
plt.close(fig)

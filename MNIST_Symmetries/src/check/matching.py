import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

PATH = '/media/osvaldo/OMV5TB/MNIST_Symmetries/'

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
    - matched_pairs: list of tuples (cluster_A_index, cluster_B_index)
    """
    # Compute contingency matrix
    cont_matrix = contingency_matrix(labels_A, labels_B)

    # Hungarian matching to maximize overlap
    cost_matrix = -cont_matrix  # maximize overlap by minimizing negative overlap
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Total overlap of matched pairs
    total_overlap = cont_matrix[row_ind, col_ind].sum()
    normalized_overlap = total_overlap / len(labels_A)

    matched_pairs = list(zip(row_ind, col_ind))

    return normalized_overlap

# =============================================================
# PARTITIONS

clusters = torch.load(PATH + 'clustering/clusters_batch_599.pth')
colors = torch.load(PATH + 'coloring/fibration_batch_599.pth')
accuracy = torch.load(PATH + 'collapse/post_training.pth')[:,5]

epsilons = clusters['eps']
thresholds = colors['L1'][:,0]
num_eps = len(epsilons)
num_thrs = len(thresholds)

matching_score = np.zeros((num_eps, num_thrs))
# =============================================================

# MATCHING IN LAYER 1.

clusters_l1 = clusters['L1']
fibers_l1 = colors['L1'][:,1:]
num_clusters = np.array([len(np.unique(row)) for row in clusters_l1])
num_colors = np.array([torch.unique(row).numel() for row in fibers_l1])

for idx_eps in range(num_eps):
	for idx_thr in range(num_thrs):

		cc = clusters_l1[idx_eps]
		ff = fibers_l1[idx_thr]
		score = one_to_one_cluster_similarity(cc, ff)
		matching_score[idx_eps, idx_thr] = score


idx_optimal_thr = np.argmax(matching_score, axis=1)
optimal_colors = num_colors[idx_optimal_thr]
optimal_thr = thresholds[idx_optimal_thr]
optimal_match = np.max(matching_score, axis=1)
optimal_acc = accuracy[idx_optimal_thr]

# Plot
fig, axs = plt.subplots(4,1, figsize =(4,20))
axs1b = axs[1].twinx()

axs[0].set_xlabel('Epsilon Synchr')
axs[1].set_xlabel('Epsilon Synchr')
axs[2].set_xlabel('Epsilon Synchr')
axs[3].set_xlabel('Epsilon Synchr')

axs[0].set_ylabel('Threshold Fibration')
axs[1].set_ylabel('Num Clusters', color = 'blue')
axs1b.set_ylabel('Num Fibers', color = 'red')
axs[2].set_ylabel('Matching')
axs[3].set_ylabel('Performance')

axs[0].plot(epsilons, optimal_thr)
axs[1].plot(epsilons, num_clusters,color='blue')
axs1b.plot(epsilons, optimal_colors, color='red')
axs[2].plot(epsilons, optimal_match)
axs[3].plot(epsilons, optimal_acc)

fig.savefig(PATH + 'plots/matching.svg', format='svg')
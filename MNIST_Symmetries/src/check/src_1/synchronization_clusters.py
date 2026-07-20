import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import numpy as np

resultsPATH = '/media/osvaldo/OMV5TB/MNIST_Symmetries/'
activities = torch.load(resultsPATH + 'activity_random_input_batch_599.pth')
hidden_size = 500 #100

#epsilons = np.linspace(0,30,100)
epsilons = np.linspace(0,2,100)
num_eps = len(epsilons)

n_clusters_layers = torch.zeros(num_eps, 3)
clusters_layers = {'L1':np.zeros((num_eps,hidden_size)),'L2':np.zeros((num_eps,hidden_size)),'L3':np.zeros((num_eps,hidden_size)), 'eps': epsilons}

for idx_eps, eps in enumerate(epsilons):
	for idx_l, h in enumerate(activities):

		h_normalized = (h - h.mean(dim=0)) / h.std(dim=0)
		h_normalized = h_normalized.cpu()
		h_normalized = h_normalized / torch.norm(h_normalized, p=2, dim=0, keepdim=True)

		distance_matrix = pairwise_distances(h_normalized.T, metric='euclidean')

		model = AgglomerativeClustering(
		    n_clusters=None,
		    metric='precomputed',
		    linkage='complete',
		    distance_threshold=eps
		)
		clusters = model.fit_predict(distance_matrix)
		n_clusters = len(np.unique(clusters))

		n_clusters_layers[idx_eps, idx_l] = n_clusters
		clusters_layers['L'+str(idx_l+1)][idx_eps,:] = clusters

torch.save(clusters_layers, resultsPATH + 'clustering/clusters_batch_599.pth')
torch.save(n_clusters_layers, resultsPATH + 'clustering/num_clusters_batch_599.pth')
from torch.nn import LSTM
from torch import cat, mm, zeros
from sklearn.cluster import AgglomerativeClustering
from torch.nn.functional import normalize
from numpy import where, unique, array, vstack

input_size  = 10
hidden_size = 20

lstm = LSTM(input_size=input_size,
			 hidden_size=hidden_size,
			 num_layers=1, batch_first=True)

params_dict = dict(lstm.named_parameters())

weight_ih = params_dict["weight_ih_l0"].detach()
weight_hh = params_dict["weight_hh_l0"].detach()

# ---------------------------------------------------

in_clusters = array([idx for idx in range(input_size)])
h_clusters  = array([0  for idx in range(hidden_size)])
c_clusters  = array([0  for idx in range(hidden_size)])

threshold = 0.7

T = 5

# ---------------------------------------------------

for t in range(T):

	print(h_clusters, c_clusters)
	
	# (1) Collapse W_ii,...W_io based on in_cluster.

	collapse_weights_i = weight_ih

	# (2) Collapse W_hi,...W_ho based on h_cluster.

	dim_out, dim_in  = weight_hh.shape
	num_h_clusters    = len(unique(h_clusters))
	collapse_weights_h = zeros((dim_out, num_h_clusters))

	for color in h_clusters:
	    indices_k = where(h_clusters == color)[0]
	    collapse_weights_h[:, color] = weight_hh[:, indices_k].sum(axis=1)


	collapse_weights = cat((collapse_weights_i,collapse_weights_h),dim=1)

	# (3) Clusters of the gates

	collapse_weights_norm = normalize(collapse_weights, dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	gates_clusters = clustering.fit_predict(distance) # I, F, G, O

	# (4) (F,C,I,G) clusters
	gates_cluster_matrix = gates_clusters.reshape(4,hidden_size)

	# (5) c_clusters
	new_c_clusters = vstack((gates_cluster_matrix[:3,:], c_clusters))
	_, c_clusters = unique(new_c_clusters.T, axis=0, return_inverse=True)

	# (6) h_clusters
	new_h_clusters = vstack((gates_cluster_matrix[3,:], c_clusters))
	_, h_clusters = unique(new_h_clusters.T, axis=0, return_inverse=True)
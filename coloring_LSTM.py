from torch.nn import LSTM
from torch import cat, mm, zeros
from sklearn.cluster import AgglomerativeClustering
from torch.nn.functional import normalize
from numpy import where, unique, array, vstack, concatenate, clip
import random

# ==========================================================

def fibration_lstm_one_layer(weight_ih, weight_hh, in_clusters, threshold):

	# (1) Collapse W_ii,...W_io based on in_cluster.

	dim_out, dim_in  = weight_ih.shape
	num_in_clusters = len(unique(in_clusters))
	collapse_weights_i = zeros((dim_out, num_in_clusters))

	for color in in_clusters:
	    indices_k = where(in_clusters == color)[0]
	    collapse_weights_i[:, color] = weight_ih[:, indices_k].sum(axis=1)

	# Dynamic

	h_clusters  = array([0  for idx in range(hidden_size)])
	c_clusters  = array([0  for idx in range(hidden_size)])

	for t in range(5):	

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
		gates_cluster_matrix = gates_clusters.reshape(4,hidden_size)

		# (5) c_clusters
		new_c_clusters = vstack((gates_cluster_matrix[:3,:], c_clusters))
		_, c_clusters = unique(new_c_clusters.T, axis=0, return_inverse=True)

		# (6) h_clusters
		new_h_clusters = vstack((gates_cluster_matrix[3,:], c_clusters))
		_, h_clusters = unique(new_h_clusters.T, axis=0, return_inverse=True)

	return gates_cluster_matrix, c_clusters, h_clusters

# ==========================================================

def opfibration_lstm_one_layer(weight_ih, weight_hh, out_clusters, threshold):
	
	input_size = weight_ih.shape[1]
	in_clusters  = array([idx  for idx in range(input_size)])
	h_clusters = out_clusters
	dim_h = len(out_clusters)

	for t in range(5):

		# (1) Collapse W_ii,...W_io based on in_cluster.

		dim_out, dim_in  = weight_ih.shape
		num_in_clusters = len(unique(in_clusters))
		collapse_weights_i = zeros((dim_out, num_in_clusters))

		for color in in_clusters:
		    indices_k = where(in_clusters == color)[0]
		    collapse_weights_i[:, color] = weight_ih[:, indices_k].sum(axis=1)		

		# (2) Collapse W_hi,...W_ho based on h_cluster.

		dim_out, dim_in  = weight_hh.shape
		num_h_clusters  = len(unique(h_clusters))
		collapse_weights_h = zeros((dim_out, num_h_clusters))

		for color in h_clusters:
		    indices_k = where(h_clusters == color)[0]
		    collapse_weights_h[:, color] = weight_hh[:, indices_k].sum(axis=1)

		collapse_weights = cat((collapse_weights_i,collapse_weights_h),dim=1)

		# (3) Clusters Fibration of the gates

		collapse_weights_norm = normalize(collapse_weights, dim=1)
		distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
		distance = distance.cpu().numpy()

		clustering = AgglomerativeClustering(
		    n_clusters=None,
		    distance_threshold=threshold,
		    linkage='average',
		    metric='precomputed')

		gates_fib_clusters = clustering.fit_predict(distance) # I, F, G, O
		gates_fib_cluster_matrix = gates_fib_clusters.reshape(4,hidden_size)

		# (4) c_clusters, gates_op_clusters
		c_clusters = vstack((gates_fib_cluster_matrix[[1,3],:], h_clusters))
		_, c_clusters = unique(c_clusters.T, axis=0, return_inverse=True)

		i_clusters = vstack((gates_fib_cluster_matrix[[1,2,3],:], h_clusters))
		_, i_clusters = unique(i_clusters.T, axis=0, return_inverse=True)

		f_clusters = vstack((gates_fib_cluster_matrix, h_clusters))
		_, f_clusters = unique(f_clusters.T, axis=0, return_inverse=True)

		g_clusters = vstack((gates_fib_cluster_matrix[[0,1,3],:], h_clusters))
		_, g_clusters = unique(g_clusters.T, axis=0, return_inverse=True)

		o_clusters = vstack((gates_fib_cluster_matrix[[0,1,2],:], h_clusters))
		_, o_clusters = unique(o_clusters.T, axis=0, return_inverse=True)

		dict_clusters = {'i':i_clusters,'f':f_clusters,'g':g_clusters,'o':o_clusters}

		# (5) gates_op_clusters

		Ws = {'i':cat((weight_ih[:dim_h,:], weight_hh[:dim_h,:]), dim=1),
			'f':cat((weight_ih[dim_h:2*dim_h,:], weight_hh[dim_h:2*dim_h,:]), dim=1),
			'g':cat((weight_ih[2*dim_h:3*dim_h,:], weight_hh[2*dim_h:3*dim_h,:]), dim=1),
			'o':cat((weight_ih[3*dim_h:4*dim_h,:], weight_hh[3*dim_h:4*dim_h,:]), dim=1)}

		all_collapse_weights = []

		for kk in ['i','f','g','o']:
			weights = Ws[kk]
			clusters = dict_clusters[kk]

			dim_out, dim_in  = weights.shape
			num_clusters = len(unique(clusters))
			collapse_weights = zeros((num_clusters, dim_in))

			for color in clusters: 
			    indices_k  = where(clusters == color)[0]
			    collapse_weights[color,:] = weights[indices_k,:].sum(axis=0)

			all_collapse_weights.append(collapse_weights)

		all_collapse_weights = cat(all_collapse_weights,dim=0)
		collapse_weights_norm = normalize(all_collapse_weights, dim=0)
		distance = 1 - mm(collapse_weights_norm.T, collapse_weights_norm)

		clustering = AgglomerativeClustering(
		    n_clusters=None,
		    distance_threshold=threshold,
		    linkage='average',
		    metric='precomputed')

		xh_clusters = clustering.fit_predict(distance)

		x_clusters = xh_clusters[:input_size]
		h_clusters = xh_clusters[input_size:]

		unique_values = unique(x_clusters)
		mapping = {val: idx for idx, val in enumerate(unique_values)}
		x_clusters = array([mapping[val] for val in x_clusters])

		unique_values = unique(h_clusters)
		mapping = {val: idx for idx, val in enumerate(unique_values)}
		h_clusters = array([mapping[val] for val in h_clusters])

	return dict_clusters, x_clusters, h_clusters

# ----------
# Test
input_size  = 10
hidden_size = 20

lstm = LSTM(input_size=input_size,
			 hidden_size=hidden_size,
			 num_layers=1, batch_first=True)

params_dict = dict(lstm.named_parameters())

weight_ih = params_dict["weight_ih_l0"].detach()
weight_hh = params_dict["weight_hh_l0"].detach()
in_clusters = array([0,0,0,1,1,2,3,4,4,5])
out_clusters = array([idx for idx in range(hidden_size)])
threshold = 0.7

#gates_clusters, c_clusters, h_clusters = fibration_lstm_one_layer(weight_ih, weight_hh, in_clusters, threshold)
dict_clusters, x_clusters, h_clusters = opfibration_lstm_one_layer(weight_ih, weight_hh, out_clusters, threshold)

print(x_clusters)

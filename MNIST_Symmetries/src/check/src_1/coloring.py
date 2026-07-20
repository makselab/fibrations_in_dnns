from sklearn.cluster import AgglomerativeClustering
from torch.nn.functional import normalize
from torch import zeros, mm, cat, tensor, where, unique

# ====================================================================

def fibration_linear(weights, in_clusters, threshold, first_layer = False, bias = None):
	dim_out, dim_in  = weights.shape

	if first_layer:
		collapse_weights = weights
	else:
		idx_in_clusters  = unique(in_clusters)
		num_in_clusters  = len(idx_in_clusters)
		collapse_weights = zeros((dim_out, num_in_clusters))

		for color in idx_in_clusters: 
		    indices_k = where(in_clusters == color)[0]
		    collapse_weights[:, color] = weights[:, indices_k].sum(axis=1)

	if bias is not None:
		dev_ = collapse_weights.device
		collapse_weights = cat((collapse_weights, bias.to(dev_).unsqueeze(1)), dim=1)

	collapse_weights_norm = normalize(collapse_weights, dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='complete',
	    metric='precomputed')

	clusters = tensor(clustering.fit_predict(distance))

	return clusters

# ====================================================================

def opfibration_linear(weights, out_clusters, threshold, last_layer = False):
	dim_out, dim_in  = weights.shape

	if last_layer:
		collapse_weights = weights
	else:
		idx_out_clusters = unique(out_clusters)
		num_out_clusters = len(idx_out_clusters)
		collapse_weights = zeros((num_out_clusters, dim_in))

		for color in idx_out_clusters: 
		    indices_k  = where(out_clusters == color)[0]
		    collapse_weights[color,:] = weights[indices_k,:].sum(axis=0)

	collapse_weights_norm = normalize(collapse_weights, dim=0)
	distance = 1 - mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='complete',
	    metric='precomputed')

	clusters = tensor(clustering.fit_predict(distance))

	return clusters

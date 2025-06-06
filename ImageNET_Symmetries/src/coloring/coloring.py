from sklearn.cluster import AgglomerativeClustering
from torch.nn.functional import normalize
from torch import zeros, mm, cat, tensor, where, unique, repeat_interleave, searchsorted
from pandas import DataFrame

# ====================================================================

def fibration_linear(weights, in_clusters, threshold, first_layer = False, bias = None):
	dim_out, dim_in  = weights.shape

	if first_layer:
		collapse_weights = weights
	else:
		idx_in_clusters  = unique(in_clusters)
		num_in_clusters  = len(idx_in_clusters)
		collapse_weights = zeros((dim_out, num_in_clusters))

		for pos_color, color in enumerate(idx_in_clusters): 
		    indices_k = where(in_clusters == color)[0]
		    collapse_weights[:, pos_color] = weights[:, indices_k].sum(axis=1)

	if bias is not None:
		dev_ = collapse_weights.device
		collapse_weights = cat((collapse_weights, bias.to(dev_).unsqueeze(1)), dim=1)

	collapse_weights_norm = normalize(collapse_weights, dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = tensor(clustering.fit_predict(distance))

	return clusters

# ====================================================================

def fibration_conv2d(weights, in_clusters, threshold, first_layer = False, bias = None):
	out_n, in_n, hx, hy = weights.shape
	weights = weights.view(out_n, in_n, -1)

	if first_layer:
		collapse_weights = weights.view(out_n, -1)
	else:
		idx_in_clusters  = unique(in_clusters)
		num_in_clusters  = len(idx_in_clusters)
		collapse_weights = zeros((out_n, num_in_clusters, hx*hy))

		for pos_color, color in enumerate(idx_in_clusters): 
		    indices_k = where(in_clusters == color)[0]
		    collapse_weights[:, pos_color, :] = weights[:, indices_k, :].sum(axis=1)

		collapse_weights = collapse_weights.view(out_n,-1)

	if bias is not None:
		dev_ = collapse_weights.device
		collapse_weights = cat((collapse_weights, bias.to(dev_).unsqueeze(1)), dim=1)

	collapse_weights_norm = normalize(collapse_weights, dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = tensor(clustering.fit_predict(distance))

	return clusters

# ====================================================================

def fibration_pooling_ch(in_clusters, times):
	return repeat_interleave(in_clusters,times)

# ====================================================================

def fibration_resblock(weights_1, weights_2, residual, in_clusters, threshold):


	# 1st Convolution of Block -------------------------
	
	interm_clusters = fibration_conv2d(weights_1, in_clusters, threshold)

	# 2nd Convolution of Block + Residual ----------------------

	out_n, in_n, hx, hy = weights_2.shape
	weights_2 = weights_2.view(out_n, in_n, -1)
	idx_interm_clusters = unique(interm_clusters)
	num_interm_clusters = len(idx_interm_clusters)
	collapse_weights_2 = zeros((out_n, num_interm_clusters, hx*hy))

	for pos_color, color in enumerate(idx_interm_clusters): 
	    indices_k = where(interm_clusters == color)[0]
	    collapse_weights_2[:, pos_color, :] = weights_2[:, indices_k, :].sum(axis=1)

	collapse_weights_2 = collapse_weights_2.view(out_n,-1)

	idx_in_clusters = unique(in_clusters)
	num_in_clusters = len(idx_in_clusters)

	if residual is not None:
		out_n, in_n, hx, hy = residual.shape
		residual = residual.view(out_n, in_n, -1)
		collapse_residual = zeros((out_n, num_in_clusters, hx*hy))

		for pos_color, color in enumerate(idx_in_clusters): 
		    indices_k = where(in_clusters == color)[0]
		    collapse_residual[:, pos_color, :] = residual[:, indices_k, :].sum(axis=1)

		collapse_residual = collapse_residual.view(out_n,-1)
	else:
		collapse_residual = zeros(len(in_clusters), num_in_clusters)
		collapse_residual.scatter_(1, tensor(in_clusters).unsqueeze(1), 1)

	collapse_weights = cat((collapse_weights_2,collapse_residual),dim=1)
	collapse_weights_norm = normalize(collapse_weights,dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = tensor(clustering.fit_predict(distance))

	return interm_clusters, clusters

# ====================================================================

def opfibration_linear(weights, out_clusters, threshold, last_layer = False):
	dim_out, dim_in  = weights.shape

	if last_layer:
		collapse_weights = weights
	else:
		idx_out_clusters = unique(out_clusters)
		num_out_clusters = len(idx_out_clusters)
		collapse_weights = zeros((num_out_clusters, dim_in))

		for pos_color, color in enumerate(idx_out_clusters): 
		    indices_k  = where(out_clusters == color)[0]
		    collapse_weights[pos_color,:] = weights[indices_k,:].sum(axis=0)

	collapse_weights_norm = normalize(collapse_weights, dim=0)
	distance = 1 - mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = tensor(clustering.fit_predict(distance))

	return clusters

# ====================================================================

def opfibration_conv2d(weights, out_clusters, threshold, last_layer = False):
	out_n, in_n, hx, hy = weights.shape
	weights = weights.view(out_n, in_n, -1)

	if last_layer:
		collapse_weights = weights.permute(0, 2, 1).reshape(-1, in_n)
	else:
		idx_out_clusters = unique(out_clusters)
		num_out_clusters = len(idx_out_clusters)	
		collapse_weights = zeros((num_out_clusters, in_n, hx*hy))

		for pos_color, color in enumerate(idx_out_clusters): 
		    indices_k = where(out_clusters == color)[0]
		    collapse_weights[pos_color, :, :] = weights[indices_k, :, :].sum(axis=0)

		collapse_weights = collapse_weights.permute(0, 2, 1).reshape(-1, in_n)

	collapse_weights_norm = normalize(collapse_weights, dim=0)
	distance = 1 - mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = tensor(clustering.fit_predict(distance))

	return clusters

# ====================================================================

def opfibration_pooling_ch(out_clusters, times):

	blocks = out_clusters.view(-1, times)
	_, clusters = unique(blocks, dim=0, return_inverse=True)

	return clusters

# ====================================================================

def opfibration_resblock(weights_1, weights_2, residual, out_clusters, threshold):

	# 2nd Convolution of Block ---------------------------

	interm_clusters = opfibration_conv2d(weights_2, out_clusters, threshold)

	# 1st Convolution of Block + Residual  ---------------------------

	out_n, in_n, hx, hy = weights_1.shape
	weights_1 = weights_1.view(out_n, in_n, -1)
	idx_interm_clusters = unique(interm_clusters)
	num_interm_clusters = len(idx_interm_clusters)
	collapse_weights_1 = zeros((num_interm_clusters, in_n, hx*hy))

	for pos_color, color in enumerate(idx_interm_clusters): 
	    indices_k = where(interm_clusters == color)[0]
	    collapse_weights_1[pos_color, :, :] = weights_1[indices_k, :,:].sum(axis=0)

	collapse_weights_1 = collapse_weights_1.permute(0, 2, 1).reshape(-1, in_n)

	idx_out_clusters = unique(out_clusters)
	num_out_clusters = len(idx_out_clusters)

	if residual is not None:
		out_n, in_n, hx, hy = residual.shape
		residual = residual.view(out_n, in_n, -1)
		collapse_residual = zeros((num_out_clusters,in_n, hx*hy))

		for pos_color, color in enumerate(idx_out_clusters): 
		    indices_k = where(out_clusters == color)[0]
		    collapse_residual[pos_color, :, :] = residual[indices_k, :, :].sum(axis=0)

		collapse_residual = collapse_residual.permute(0, 2, 1).reshape(-1, in_n)
	else:
		collapse_residual = zeros(num_out_clusters, len(out_clusters))
		collapse_residual.scatter_(0, tensor(out_clusters).unsqueeze(0), 1)

	collapse_weights = cat((collapse_weights_1,collapse_residual),dim=0)
	collapse_weights_norm = normalize(collapse_weights,dim=0)
	distance = 1 - mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = tensor(clustering.fit_predict(distance))

	return interm_clusters, clusters

# ====================================================================

def covering_layer(df_fibration, df_opfibration):
	'''
	df_fibration, df_opfibration  = N x Times
	'''

	df_cov = DataFrame(index=df_fibration.index, columns=df_fibration.columns)

	for n in df_fibration.index:
		for t in df_fibration.columns:
			df_cov.at[n, t] = (df_fibration.at[n, t], df_opfibration.at[n, t])

	return df_cov

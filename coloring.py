from sklearn.cluster import AgglomerativeClustering
from torch.nn.functional import normalize
from numpy import where, unique
from torch import zeros, mm, cat, tensor

# ====================================================================

def fibration_linear(weights, in_clusters, threshold, first_layer = False, bias=None):
	dim_out, dim_in  = weights.shape

	if first_layer:
		collapse_weights = weights
	else:	
		num_in_clusters  = len(unique(in_clusters))
		collapse_weights = zeros((dim_out, num_in_clusters))

		for color in in_clusters: 
		    indices_k = where(in_clusters == color)[0]
		    collapse_weights[:, color] = weights[:, indices_k].sum(axis=1)

	if bias is not None:
		collapse_weights = cat((colapse_weight, bias), dim=1)

	collapse_weights_norm = normalize(collapse_weights, dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = clustering.fit_predict(distance)

	return clusters

# ====================================================================

def fibration_conv2d(weights, in_clusters, threshold, first_layer = False):
	out_n, in_n, hx, hy = weights.shape
	weights = weights.view(out_n, in_n, -1)

	if first_layer:
		collapse_weights = weights.view(out_n, -1)
	else:
		num_in_clusters  = len(unique(in_clusters))
		collapse_weights = zeros((out_n, num_in_clusters, hx*hy))

		for color in in_clusters: 
		    indices_k = where(in_clusters == color)[0]
		    collapse_weights[:, color, :] = weights[:, indices_k, :].sum(axis=1)

		collapse_weights = collapse_weights.view(out_n,-1)

	collapse_weights_norm = normalize(collapse_weights, dim=1)
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = clustering.fit_predict(distance)

	return clusters

# ====================================================================

def fibration_resblock(weights_1, weights_2, residual, in_clusters, threshold):


	# 1st Convolution of Block -------------------------
	
	interm_clusters = fibration_conv2d(weights_1, in_clusters, threshold)

	# 2nd Convolution of Block + Residual ----------------------

	out_n, in_n, hx, hy = weights_2.shape
	weights_2 = weights_2.view(out_n, in_n, -1)
	num_interm_clusters = len(unique(interm_clusters))
	collapse_weights_2 = zeros((out_n, num_interm_clusters, hx*hy))

	for color in interm_clusters: 
	    indices_k = where(interm_clusters == color)[0]
	    collapse_weights_2[:, color, :] = weights_2[:, indices_k, :].sum(axis=1)

	collapse_weights_2 = collapse_weights_2.view(out_n,-1)

	num_in_clusters = len(unique(in_clusters))

	if residual is not None:
		out_n, in_n, hx, hy = residual.shape
		residual = residual.view(out_n, in_n, -1)
		collapse_residual = zeros((out_n, num_in_clusters, hx*hy))

		for color in in_clusters: 
		    indices_k = where(in_clusters == color)[0]
		    collapse_residual[:, color, :] = residual[:, indices_k, :].sum(axis=1)

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

	clusters = clustering.fit_predict(distance)

	return interm_clusters, clusters

# ====================================================================

def opfibration_linear(weights, out_clusters, threshold, last_layer = False):
	dim_out, dim_in  = weights.shape

	if last_layer:
		collapse_weights = weights
	else:
		num_out_clusters = len(unique(out_clusters))
		collapse_weights = zeros((num_out_clusters, dim_in))

		for color in out_clusters: 
		    indices_k  = where(out_clusters == color)[0]
		    collapse_weights[color,:] = weights[indices_k,:].sum(axis=0)

	collapse_weights_norm = normalize(collapse_weights, dim=0)
	distance = 1 - mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = clustering.fit_predict(distance)

	return clusters

# ====================================================================

def opfibration_conv2d(weights, out_clusters, threshold, last_layer = False):
	out_n, in_n, hx, hy = weights.shape
	weights = weights.view(out_n, in_n, -1)

	if last_layer:
		collapse_weights = weights.permute(0, 2, 1).reshape(-1, in_n)
	else:
		num_out_clusters = len(unique(out_clusters))	
		collapse_weights = zeros((num_out_clusters, in_n, hx*hy))

		for color in out_clusters: 
		    indices_k = where(out_clusters == color)[0]
		    collapse_weights[color, :, :] = weights[indices_k, :, :].sum(axis=0)

		collapse_weights = collapse_weights.permute(0, 2, 1).reshape(-1, in_n)

	collapse_weights_norm = normalize(collapse_weights, dim=0)
	distance = 1 - mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	clustering = AgglomerativeClustering(
	    n_clusters=None,
	    distance_threshold=threshold,
	    linkage='average',
	    metric='precomputed')

	clusters = clustering.fit_predict(distance)

	return clusters

# ====================================================================

def opfibration_resblock(weights_1, weights_2, residual, out_clusters, threshold):

	# 2nd Convolution of Block ---------------------------

	interm_clusters = opfibration_conv2d(weights_2, out_clusters, threshold)

	# 1st Convolution of Block + Residual  ---------------------------

	out_n, in_n, hx, hy = weights_1.shape
	weights_1 = weights_1.view(out_n, in_n, -1)
	num_interm_clusters = len(unique(interm_clusters))
	collapse_weights_1 = zeros((num_interm_clusters, in_n, hx*hy))

	for color in interm_clusters: 
	    indices_k = where(interm_clusters == color)[0]
	    collapse_weights_1[color, :, :] = weights_1[indices_k, :,:].sum(axis=0)

	collapse_weights_1 = collapse_weights_1.permute(0, 2, 1).reshape(-1, in_n)

	num_out_clusters = len(unique(out_clusters))

	if residual is not None:
		out_n, in_n, hx, hy = residual.shape
		residual = residual.view(out_n, in_n, -1)
		collapse_residual = zeros((num_out_clusters,in_n, hx*hy))

		for color in out_clusters: 
		    indices_k = where(out_clusters == color)[0]
		    collapse_residual[color, :, :] = residual[indices_k, :, :].sum(axis=0)

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

	clusters = clustering.fit_predict(distance)

	return interm_clusters, clusters

# ====================================================================

def covering_layer(df_fibration, df_opfibration):
	'''
	df_fibration, df_opfibration  = N x Times
	'''

	for n in df_fibrations.index:
		for t in df_fibrations.columns:
			df_cov.at[n, t] = (df_fibrations.at[n, t], df_optfibrations.at[n, t])

	return df_cov

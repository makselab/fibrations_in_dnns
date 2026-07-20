from torch import unique, zeros, where, cat, tensor, is_tensor, stack, mm, column_stack
from torch.nn.functional import normalize
from .clustering import make_algorithm

# ==============================================================================
# ==============================================================================

def fibration_linear(weights, bias, in_colors, clustering_method, distance_thr):

	"""
	Args
	----------
	weights : torch.Tensor (2D)
	    Weight tensor of shape (m x n)

	bias : torch.Tensor (1D) or None
	    Bias tensor of shape (m,)

	in_colors : torch.Tensor (1D)
	    Contains colors indices for each of the n elements.

	clustering_method : dict
	    Clustering method configuration dictionary with keys:
	    - 'name': str, method name
	    - 'cfg': dict, method-specific parameters (e.g 'linkage')


	distance_thr : float
		Distance threshold in range [0, 2] for deciding cluster merging.
		- 0: theoretical definition of fibrations
		- 2: all nodes are in the same cluster


	Returns
	-------
	out_colors : torch.Tensor (1D)
		Contains colors indices for each of the m elements.

	Notes
	-----

	"""

	# Check args =================================================
	assert is_tensor(weights), "weights must be a PyTorch tensor"
	assert is_tensor(in_colors), "in_colors must be a PyTorch tensor"

	assert weights.dim()   == 2, f"weights must be 2D tensor, got {weights.dim()}D"
	assert in_colors.dim() == 1, f"in_colors must be 1D tensor, got {in_colors.dim()}D"

	dim_out, dim_in  = weights.shape
	device =  weights.device

	# Collapsed Weights based on In_Clusters ===================
	assert dim_in == in_colors.size(0), f"Dimension mismatch: weights has {dim_in} columns, in_colors has {in_colors.size(0)} elements"

	idx_in_colors    = unique(in_colors)
	collapse_weights = stack([weights[:, in_colors == color].sum(dim=1) for color in idx_in_colors], dim=1)

	if bias is not None:
		assert is_tensor(bias), "bias must be a PyTorch tensor"
		assert bias.dim() == 1, f"bias must be 1D tensor, got {bias.dim()}D"
		new_row = bias.unsqueeze(1).to(device)
		collapse_weights = cat((collapse_weights, new_row), dim=1)

	# Normalization of Collapsed Weights =======================
	collapse_weights_norm = normalize(collapse_weights, dim=1)

	# Distance Matrix ==========================================
	distance = 1 - mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.clamp(min=0)
	distance = distance.detach().cpu().numpy()

	# Clustering of Distance Matrix ============================
	assert 0 <= distance_thr <= 2, f"distance_thr must be between 0 and 2, got {distance_thr}"

	algorithm = make_algorithm(clustering_method['name'],clustering_method['cfg'], distance_thr)
	out_colors = algorithm(distance)
	out_colors = tensor(out_colors)

	return out_colors

# ==============================================================================
# ==============================================================================

def opfibration_linear(weights, bias, out_colors, clustering_method, distance_thr):

	"""
	Args
	----------
	weights : torch.Tensor (2D)
	    Weight tensor of shape (m x n)

	bias : torch.Tensor (1D) or None
	    Bias tensor of shape (m,)

	out_colors : torch.Tensor (1D)
	    Contains colors indices for each of the m elements.

	clustering_method : dict
	    Clustering method configuration dictionary with keys:
	    - 'name': str, method name
	    - 'cfg': dict, method-specific parameters (e.g 'linkage')


	distance_thr : float
		Distance threshold in range [0, 2] for deciding cluster merging.
		- 0: theoretical definition of fibrations
		- 2: all nodes are in the same cluster


	Returns
	-------
	in_colors : torch.Tensor (1D)
		Contains colors indices for each of the n elements.

	Notes
	-----

	"""

	# Check args =================================================
	assert is_tensor(weights), "weights must be a PyTorch tensor"
	assert is_tensor(out_colors), "in_colors must be a PyTorch tensor"

	assert weights.dim()   == 2, f"weights must be 2D tensor, got {weights.dim()}D"
	assert out_colors.dim() == 1, f"out_colors must be 1D tensor, got {in_colors.dim()}D"

	dim_out, dim_in  = weights.shape
	device =  weights.device

	# Collapsed Weights based on In_Clusters ===================
	assert dim_out == out_colors.size(0), f"Dimension mismatch: weights has {dim_out} columns, in_colors has {out_colors.size(0)} elements"

	idx_out_colors    = unique(out_colors)
	num_out_colors    = len(idx_out_colors)
	collapse_weights = stack([weights[out_colors == color,:].sum(dim=0) for color in idx_out_colors], dim=0)

	# Normalization of Collapsed Weights =======================
	collapse_weights_norm = normalize(collapse_weights, dim=0)

	# Distance Matrix ==========================================
	distance = 1 - mm(collapse_weights_norm.T, collapse_weights_norm)
	distance = distance.cpu().numpy()

	# Clustering of Distance Matrix ============================
	assert 0 <= distance_thr <= 2, f"distance_thr must be between 0 and 2, got {distance_thr}"

	algorithm  = make_algorithm(clustering_method['name'],clustering_method['cfg'], distance_thr)
	in_colors = algorithm(distance)
	in_colors = tensor(in_colors)

	return in_colors

# ==============================================================================
# ==============================================================================

def covering(fibration_colors, opfibration_colors):
	color_pairs = column_stack((fibration_colors, opfibration_colors))  # shape: (n, 2)
	_, covering = unique(color_pairs, dim=0, return_inverse=True)

	return covering

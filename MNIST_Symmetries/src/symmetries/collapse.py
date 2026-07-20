import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch import cat, eye

def collapse_linear(module, collapse_in=True, collapse_out=True):
    weights = module.weight.data
    bias    = module.bias.data if module.bias is not None else None
    dim_out, dim_in = weights.shape
    device  = module.weight.device
    dtype   = module.weight.dtype

    # Collapse output dimension
    if collapse_out:
        num_out_colors     = torch.unique(module.out_colors).shape[0]
        out_mtx_partition  = torch.zeros(num_out_colors, dim_out).scatter_(0, module.out_colors.unsqueeze(0), 1).to(device)
        out_sizes_clusters = out_mtx_partition.sum(dim=1, keepdim=True)  # (num_out_colors, 1)
        W_coll             = (out_mtx_partition @ weights) / out_sizes_clusters        # (num_out_colors, num_in_colors)
        bias_coll          = (out_mtx_partition @ bias) / out_sizes_clusters.squeeze() if bias is not None else None
    else:
        num_out_colors = dim_out
        W_coll         = weights
        bias_coll      = bias

    dW = W_coll[module.out_colors] - weights
    db = bias_coll[module.out_colors] - bias

    # Collapse input dimension
    if collapse_in:
        num_in_colors      = torch.unique(module.in_colors).shape[0]
        in_mtx_partition   = torch.zeros(num_in_colors, dim_in).scatter_(0, module.in_colors.unsqueeze(0), 1).to(device)
        W_coll = W_coll @ in_mtx_partition.T        # (dim_out, num_in_colors)
    else:
        num_in_colors = dim_in

    # Build collapsed module
    module_coll             = nn.Linear(num_in_colors, num_out_colors, bias=bias is not None, device=device, dtype=dtype)
    module_coll.weight.data = W_coll

    if bias_coll is not None:
        module_coll.bias.data = bias_coll
    else:
        module_coll.bias = None

    return module_coll, dW, db
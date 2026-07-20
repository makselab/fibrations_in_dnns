import torch.nn as nn
import torch 
import torch.nn.functional as F
from torch import cat, eye

def ablation_linear(module, nodes_ablation_in=None, nodes_ablation_out=None):
    weights = module.weight.data
    bias    = module.bias.data if module.bias is not None else None
    dim_out, dim_in = weights.shape
    device  = module.weight.device
    dtype   = module.weight.dtype

    W_abl = weights
    b_abl = bias if bias is not None else None
    num_in_nodes = dim_in
    num_out_nodes = dim_out

    if nodes_ablation_in is not None:
        W_abl = W_abl[:,nodes_ablation_in]
        num_in_nodes = len(nodes_ablation_in)

    if nodes_ablation_out is not None:
        W_abl = W_abl[nodes_ablation_out]
        b_abl = b_abl[nodes_ablation_out] if b_abl is not None else None
        num_out_nodes = len(nodes_ablation_out)

    # Build collapsed module
    module_ablation         = nn.Linear(num_in_nodes, num_out_nodes, bias=bias is not None, device=device, dtype=dtype)
    module_ablation.weight.data = W_abl

    if b_abl is not None:
        module_ablation.bias.data = b_abl
    else:
        module_ablation.bias = None

    return module_ablation
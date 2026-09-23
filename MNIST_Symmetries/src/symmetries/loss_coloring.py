import torch
import numpy as np
from torch import tensor, is_tensor

from .clustering import make_algorithm
from .dL_matrix import compute_dL, compute_dL_v2

# ==============================================================================
# ==============================================================================

def loss_coloring_linear(weights, bias, S, A, clustering_method, distance_thr, dL_fn=compute_dL_v2):

    assert is_tensor(weights), "weights must be a PyTorch tensor"
    assert is_tensor(bias),    "bias must be a PyTorch tensor"
    assert weights.dim() == 2, f"weights must be 2D, got {weights.dim()}D"
    assert bias.dim()    == 1, f"bias must be 1D, got {bias.dim()}D"

    w = torch.cat([weights, bias.unsqueeze(1)], dim=1)  # (n_out, n_in+1)

    dL    = dL_fn(w, A, S)
    dL_np = dL.detach().cpu().numpy()

    dL_max = dL_np.max()
    if dL_max > 0:
        dL_np = dL_np / dL_max

    algorithm  = make_algorithm(clustering_method['name'], clustering_method['cfg'], distance_thr)
    out_colors = algorithm(dL_np)

    return tensor(out_colors)

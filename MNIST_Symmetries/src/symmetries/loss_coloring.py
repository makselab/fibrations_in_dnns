import torch
import numpy as np
from torch import tensor, is_tensor

from .clustering import make_algorithm
from .dL_matrix import compute_dL

# ==============================================================================
# ==============================================================================

def loss_coloring_linear(weights, bias, S, A, clustering_method, distance_thr):

    """
    Args
    ----------
    weights : torch.Tensor (2D)
        Weight tensor of shape (n_out, n_in).

    bias : torch.Tensor (1D)
        Bias tensor of shape (n_out,).

    S : torch.Tensor (2D)
        Gradient covariance matrix of shape (n_out, n_out).
        S_ij = E[grad_z_i * grad_z_j].

    A : torch.Tensor (2D)
        Augmented activation covariance of shape (n_in+1, n_in+1).
        A = E[a_aug a_aug^T] where a_aug = [a | 1].

    clustering_method : dict
        Clustering method configuration with keys:
        - 'name': str, 'linkage_fcluster' or 'agg_clustering'
        - 'cfg':  dict, e.g. {'linkage': 'average'}

    distance_thr : float
        Distance threshold for cluster merging.
        - 0   : no merges
        - inf : all nodes in one cluster

    Returns
    -------
    out_colors : torch.Tensor (1D)
        Color index for each of the n_out output nodes.

    Notes
    -----
    dL(i,j) = 1/2 * P(i,j) * D(i,j)

      P(i,j) = (S_ii S_jj - S_ij^2) / (S_ii + S_jj + 2 S_ij)
      D(i,j) = (w_i - w_j)^T A (w_i - w_j)

    dL >= 0 always: P >= 0 by Cauchy-Schwarz (S is PSD);
    denominator >= 0 by AM-GM.
    """

    # Check args =================================================
    assert is_tensor(weights), "weights must be a PyTorch tensor"
    assert is_tensor(bias),    "bias must be a PyTorch tensor"
    assert weights.dim() == 2, f"weights must be 2D, got {weights.dim()}D"
    assert bias.dim()    == 1, f"bias must be 1D, got {bias.dim()}D"

    # Build augmented weight matrix w = [W | b] ================
    w = torch.cat([weights, bias.unsqueeze(1)], dim=1)  # (n_out, n_in+1)

    # Distance Matrix (torch, stays on device) =================
    dL    = compute_dL(w, A, S)                          # (n_out, n_out) torch
    dL_np = dL.detach().cpu().numpy()

    # Normalize to [0, 1] so distance_threshold is scale-independent
    dL_max = dL_np.max()
    if dL_max > 0:
        dL_np = dL_np / dL_max

    # Clustering (sklearn expects numpy) =======================
    algorithm  = make_algorithm(clustering_method['name'], clustering_method['cfg'], distance_thr)
    out_colors = algorithm(dL_np)

    return tensor(out_colors)

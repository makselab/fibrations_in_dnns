import torch
import torch.nn as nn
import numpy as np

def collapse_linear(module, S, collapse_in=True, collapse_out=True):
    """
    Output collapse  : W_{c,k} = (sum_{i,j in c} S_{ij} w_{i,k}) / (sum_{i,j in c} S_{ij})
    Input  collapse  : sum over input-cluster nodes  (no normalization)

    Args
    ----
    module : nn.Linear with module.in_colors / module.out_colors set
    S      : np.ndarray (dim_out, dim_out) — gradient covariance for output nodes
    """
    weights = module.weight.data
    bias    = module.bias.data if module.bias is not None else None
    dim_out, dim_in = weights.shape
    device  = module.weight.device
    dtype   = module.weight.dtype

    # ------------------------------------------------------------------
    # Collapse output dimension (S-weighted average of rows)
    # ------------------------------------------------------------------
    if collapse_out:
        out_colors     = module.out_colors.cpu().numpy()
        num_out_colors = int(np.unique(out_colors).shape[0])
        W_np = weights.cpu().numpy()                                 # (dim_out, dim_in)
        b_np = bias.cpu().numpy() if bias is not None else None

        W_coll = np.zeros((num_out_colors, dim_in))
        b_coll = np.zeros(num_out_colors) if b_np is not None else None

        for c in range(num_out_colors):
            c_idx = np.where(out_colors == c)[0]
            S_c   = S[np.ix_(c_idx, c_idx)]      # principal submatrix (PSD)
            r     = S_c.sum(axis=1)              # r_i = sum_{j in c} S_{ij}
            M     = r.sum()                      # sum_{i,j in c} S_{ij} >= 0
            if M < 1e-15:
                r = np.ones(len(c_idx)); M = float(len(c_idx))
            W_coll[c] = (r @ W_np[c_idx]) / M
            if b_np is not None:
                b_coll[c] = (r @ b_np[c_idx]) / M

        W_coll = torch.tensor(W_coll, dtype=dtype, device=device)
        b_coll = torch.tensor(b_coll, dtype=dtype, device=device) if b_coll is not None else None
    else:
        num_out_colors = dim_out
        W_coll = weights
        b_coll = bias

    dW = W_coll[module.out_colors] - weights
    db = b_coll[module.out_colors] - bias if bias is not None else None

    # ------------------------------------------------------------------
    # Collapse input dimension (sum over input-cluster nodes)
    # ------------------------------------------------------------------
    if collapse_in:
        in_colors        = module.in_colors.to(device)
        num_in_colors    = torch.unique(in_colors).shape[0]
        in_mtx_partition = torch.zeros(num_in_colors, dim_in, device=device, dtype=dtype)
        in_mtx_partition.scatter_(0, in_colors.unsqueeze(0), 1)
        W_coll = W_coll @ in_mtx_partition.T     # sum (no normalization)
    else:
        num_in_colors = dim_in

    # ------------------------------------------------------------------
    # Build collapsed module
    # ------------------------------------------------------------------
    module_coll             = nn.Linear(num_in_colors, num_out_colors, bias=bias is not None, device=device, dtype=dtype)
    module_coll.weight.data = W_coll
    if b_coll is not None:
        module_coll.bias.data = b_coll
    else:
        module_coll.bias = None

    return module_coll, dW, db


def collapse_linear_v2(module, S, collapse_in=True, collapse_out=True):
    """
    Output collapse  : W_coll = (P^T S P)^{-1} P^T S W
    Input  collapse  : sum over input-cluster nodes  (no normalization)

    Args
    ----
    module : nn.Linear with module.in_colors / module.out_colors set
    S      : np.ndarray (dim_out, dim_out) — gradient covariance for output nodes
    """
    weights = module.weight.data
    bias    = module.bias.data if module.bias is not None else None
    dim_out, dim_in = weights.shape
    device  = module.weight.device
    dtype   = module.weight.dtype

    # ------------------------------------------------------------------
    # Collapse output dimension
    # ------------------------------------------------------------------
    if collapse_out:
        out_colors     = module.out_colors.to(device)
        num_out_colors = int(torch.unique(out_colors).shape[0])

        S_t = torch.tensor(S, dtype=dtype, device=device)       # (dim_out, dim_out)

        # Membership matrix P: (dim_out, num_out_colors)
        P = torch.zeros(dim_out, num_out_colors, dtype=dtype, device=device)
        P.scatter_(1, out_colors.unsqueeze(1), 1.0)

        PtS   = P.t() @ S_t                                      # (num_colors, dim_out)
        PtSP  = PtS @ P                                          # (num_colors, num_colors)
        PtSW  = PtS @ weights                                    # (num_colors, dim_in)
        W_coll = torch.linalg.pinv(PtSP) @ PtSW                 # (num_colors, dim_in)

        if bias is not None:
            b_coll = torch.linalg.pinv(PtSP) @ (PtS @ bias)    # (num_colors,)
        else:
            b_coll = None
    else:
        num_out_colors = dim_out
        W_coll = weights
        b_coll = bias

    dW = W_coll[module.out_colors] - weights
    db = b_coll[module.out_colors] - bias if bias is not None else None

    # ------------------------------------------------------------------
    # Collapse input dimension (sum over input-cluster nodes)
    # ------------------------------------------------------------------
    if collapse_in:
        in_colors        = module.in_colors.to(device)
        num_in_colors    = torch.unique(in_colors).shape[0]
        in_mtx_partition = torch.zeros(num_in_colors, dim_in, device=device, dtype=dtype)
        in_mtx_partition.scatter_(0, in_colors.unsqueeze(0), 1)
        W_coll = W_coll @ in_mtx_partition.T
    else:
        num_in_colors = dim_in

    # ------------------------------------------------------------------
    # Build collapsed module
    # ------------------------------------------------------------------
    module_coll             = nn.Linear(num_in_colors, num_out_colors, bias=bias is not None, device=device, dtype=dtype)
    module_coll.weight.data = W_coll
    if b_coll is not None:
        module_coll.bias.data = b_coll
    else:
        module_coll.bias = None

    return module_coll, dW, db
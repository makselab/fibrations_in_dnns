import torch

def compute_dL(W, A, S, s_reg=0.01, eps=1e-12):
    """
    W: torch.Tensor (n_out, n_in+1)  augmented weight matrix [W | b]
    A: torch.Tensor (n_in+1, n_in+1) activation covariance
    S: torch.Tensor (n_out, n_out)   gradient covariance
    s_reg: diagonal regularization added to S as fraction of mean(diag(S))
           prevents P from collapsing to 0 when S is low-rank
    """

    G = W @ A @ W.t()                               # (n_out, n_out)  w_i^T A w_j
    g = torch.diag(G)                                # (n_out,)
    D = g[:, None] + g[None, :] - 2 * G             # (w_i - w_j)^T A (w_i - w_j)

    s   = torch.diag(S)                          # (n_out,)
    num = s[:, None] * s[None, :] - S ** 2
    den = s[:, None] + s[None, :] + 2 * S
    P   = num / (den + eps)

    dL = 0.5 * P * D
    dL.fill_diagonal_(0.0)
    return dL


def compute_dL_v2(W, A, S, s_reg=1e-3, eps=1e-12):
    """
    W: torch.Tensor (n_out, n_in+1)  augmented weight matrix [W | b]
    A: torch.Tensor (n_in+1, n_in+1) activation covariance
    S: torch.Tensor (n_out, n_out)   gradient covariance
    s_reg: regularization added to S as fraction of mean(diag(S)) before inverting
           ensures Z = (S + s_reg * mean(diag(S)) * I)^{-1} is positive definite

    P(i,j) = 1 / (Z_ii + Z_jj - 2*Z_ij)   where Z = inv(S_reg)
    """

    G = W @ A @ W.t()
    g = torch.diag(G)
    D = g[:, None] + g[None, :] - 2 * G

    s_mean = torch.diag(S).mean().clamp(min=1e-15)
    S_reg  = (1-s_reg) * S + s_reg * s_mean * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    Z      = torch.linalg.inv(S_reg)

    z     = torch.diag(Z)
    denom = z[:, None] + z[None, :] - 2 * Z
    P     = 1.0 / (denom.clamp(min=eps))

    dL = 0.5 * P * D
    dL.fill_diagonal_(0.0)
    return dL

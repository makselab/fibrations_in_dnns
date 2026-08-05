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

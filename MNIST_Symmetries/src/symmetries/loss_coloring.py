"""
Multi-layer clustering with a shared F_max budget
==========================================================================

dL = dL_1(c_1) + dL_2(c_2) + ... + dL_l(c_l)
Constraint: F_total < F_max  (one shared budget across all layers)

Because the merge cost (delta) can be negative (non-monotone), the correct
strategy is a single heap mixing candidates from ALL layers, with a single
shared F_total accumulator. At each step the cheapest available merge across
all layers is applied (regardless of sign), and the loop stops as soon as
the cheapest candidate would exceed the total budget.

Performance note: instead of computing rep @ A @ rep[k] (O(d^2)) on every
merge_delta call, we maintain Aw[k] = A @ rep[k] for each active cluster k.
This reduces each merge_delta from O(neighbors * d^2) to O(neighbors * d).
Aw is updated in O(d) per merge using the precomputed Aloc/Across vectors,
which track A @ b_local and A @ b_cross respectively.
"""

import heapq
import itertools
import numpy as np
import scipy.sparse as sp


def build_initial_structures(w, S, A):
    num_nodes, feature_dim = w.shape

    if not sp.issparse(S):
        S = sp.csr_matrix(S)
    S = S.tocsr()

    # b_global[i] = (S @ w)[i]: weighted sum of neighbor features seen by node i
    weighted_features = S @ w
    M_local  = {i: 0.0                         for i in range(num_nodes)}
    b_local  = {i: np.zeros(feature_dim)       for i in range(num_nodes)}
    b_global = {i: weighted_features[i].copy() for i in range(num_nodes)}
    rep      = {i: w[i].copy()                 for i in range(num_nodes)}
    M_cross  = {i: {}                          for i in range(num_nodes)}
    b_cross  = {i: {}                          for i in range(num_nodes)}

    # Parse sparse adjacency into local (diagonal) and cross (off-diagonal) terms
    S_coo = S.tocoo()
    for dst, src, val in zip(S_coo.row, S_coo.col, S_coo.data):
        if val == 0.0:
            continue
        if dst == src:
            M_local[dst] += val
            b_local[dst] += val * w[src]
        else:
            M_cross[dst][src] = M_cross[dst].get(src, 0.0) + val
            b_cross[dst][src] = b_cross[dst].get(src, np.zeros(feature_dim)) + val * w[src]

    nodes = {i: {i} for i in range(num_nodes)}
    return (M_local, b_local, b_global, rep, M_cross, b_cross, nodes)


def _F_diag(mass_k, rep_k, b_global_k, Aw_k):
    """
    Diagonal term of the objective F for cluster k.
    Aw_k = A @ rep_k (precomputed); avoids O(d^2) matrix products.
    """
    return float(mass_k * np.dot(Aw_k, rep_k) - 2.0 * np.dot(Aw_k, b_global_k))


def merge_delta(a, b, M_local, b_local, b_global, rep, M_cross, b_cross, Aw, Aloc, Across):
    """
    Compute the change in F (delta) from merging clusters a and b,
    and return the state of the resulting merged cluster c.

    All A-weighted inner products are computed as O(d) dot products using
    Aw[k] = A @ rep[k], Aloc[k] = A @ b_local[k], Across[k][j] = A @ b_cross[k][j].
    """
    feature_dim = b_global[a].shape[0]
    zero = np.zeros(feature_dim)

    # Cross-connection mass and directed feature vectors between a and b
    M_ab = M_cross[a].get(b, 0.0)
    b_ab = b_cross[a].get(b, zero)
    b_ba = b_cross[b].get(a, zero)

    # State of the merged cluster c
    M_c        = M_local[a] + M_local[b] + 2 * M_ab
    b_local_c  = b_local[a] + b_local[b] + b_ab + b_ba
    b_global_c = b_global[a] + b_global[b]

    # Aw_c = A @ rep_c derived from Aloc (O(d), no matrix multiply)
    Aloc_c = Aloc[a] + Aloc[b] + Across[a].get(b, zero) + Across[b].get(a, zero)
    if M_c > 1e-15:
        rep_c = b_local_c / M_c
        Aw_c  = Aloc_c / M_c
    else:
        rep_c = 0.5 * (rep[a] + rep[b])
        Aw_c  = 0.5 * (Aw[a] + Aw[b])

    neighbors = (set(M_cross[a]) | set(M_cross[b])) - {a, b}

    # Cost before merge: diagonal terms of a and b plus their cross-interaction
    F_before = (_F_diag(M_local[a], rep[a], b_global[a], Aw[a]) + _F_diag(M_local[b], rep[b], b_global[b], Aw[b]) + 2 * M_ab * np.dot(Aw[a], rep[b]))
    for k in neighbors:
        M_ak = M_cross[a].get(k, 0.0)
        M_bk = M_cross[b].get(k, 0.0)
        F_before += 2 * M_ak * np.dot(Aw[a], rep[k]) + 2 * M_bk * np.dot(Aw[b], rep[k])

    # Cost after merge: diagonal term of c plus its cross-interactions with neighbors
    F_after = _F_diag(M_c, rep_c, b_global_c, Aw_c)
    for k in neighbors:
        M_ck = M_cross[a].get(k, 0.0) + M_cross[b].get(k, 0.0)
        F_after += 2 * M_ck * np.dot(Aw_c, rep[k])

    delta = F_after - F_before
    return delta, rep_c, M_c, b_local_c, b_global_c, Aw_c, Aloc_c


# ------------------------------------------------------------------
# Multi-layer clustering with shared budget
# ------------------------------------------------------------------

class LayerState:
    """Holds the mutable clustering state for one layer."""
    __slots__ = ("M_local", "b_local", "b_global", "rep", "M_cross",
                 "b_cross", "nodes", "active", "new_id_gen",
                 "Aw", "Aloc", "Across")

    def __init__(self, w, S, A):
        (self.M_local, self.b_local, self.b_global, self.rep,
         self.M_cross, self.b_cross, self.nodes) = build_initial_structures(w, S, A)

        self.active     = set(self.nodes.keys())
        self.new_id_gen = itertools.count(max(self.active) + 1 if self.active else 0)

        # Batch precompute Aw[j] = A @ w[j] for all initial nodes (one BLAS call)
        Aw_matrix = w @ A                                        # (n, d), uses A symmetric
        self.Aw   = {j: Aw_matrix[j].copy() for j in range(w.shape[0])}

        # Aloc[j] = A @ b_local[j].  Initially b_local[j] = M_local[j] * w[j]
        # so Aloc[j] = M_local[j] * Aw[j].
        self.Aloc = {j: self.M_local[j] * self.Aw[j] for j in range(w.shape[0])}

        # Across[j][k] = A @ b_cross[j][k].  Initially b_cross[j][k] = M_cross[j][k] * w[k]
        # so Across[j][k] = M_cross[j][k] * Aw[k].
        self.Across = {
            j: {k: self.M_cross[j][k] * self.Aw[k] for k in self.M_cross[j]}
            for j in range(w.shape[0])
        }


def cluster_multilayer_shared_budget(layers, F_max_total, verbose=True):
    """
    layers      : list of dicts {"w": ..., "S": ..., "A": ...}
    F_max_total : shared budget — merging stops when F_total would exceed it

    Returns: list of (clusters, representatives) per layer, and final F_total.
    """
    # Initialize one state object per layer
    states = [LayerState(layer["w"], layer["S"], layer["A"]) for layer in layers]

    heap = []  # entries: (delta, layer_idx, node_a, node_b)

    def push_candidates_for(layer_idx, k):
        """Push all valid merge candidates involving node k in the given layer."""
        state = states[layer_idx]
        for j in list(state.M_cross[k].keys()):
            if j == k or j not in state.active:
                continue
            node_a, node_b = (k, j) if k < j else (j, k)
            delta, *_ = merge_delta(node_a, node_b,
                                    state.M_local, state.b_local, state.b_global,
                                    state.rep, state.M_cross, state.b_cross,
                                    state.Aw, state.Aloc, state.Across)
            heapq.heappush(heap, (delta, layer_idx, node_a, node_b))

    # Seed the heap with all candidate merges across all layers
    for layer_idx, state in enumerate(states):
        for k in list(state.active):
            push_candidates_for(layer_idx, k)

    F_total     = 0.0
    merge_count = 0

    while heap:
        _, layer_idx, a, b = heapq.heappop(heap)
        state = states[layer_idx]

        # Discard stale entries (one or both nodes were already merged)
        if a not in state.active or b not in state.active:
            continue

        # Recompute delta with current state and check against budget
        (delta, rep_c, M_c, b_local_c, b_global_c,
         Aw_c, Aloc_c) = merge_delta(a, b,
                                      state.M_local, state.b_local, state.b_global,
                                      state.rep, state.M_cross, state.b_cross,
                                      state.Aw, state.Aloc, state.Across)

        if F_total + delta > F_max_total:
            break

        # Create merged cluster c and update its cross-connections to neighbors
        c = next(state.new_id_gen)
        neighbors = (set(state.M_cross[a]) | set(state.M_cross[b])) - {a, b}

        zero_d = np.zeros(rep_c.shape[0])

        state.M_cross[c]  = {}
        state.b_cross[c]  = {}
        state.Across[c]   = {}
        for k in neighbors:
            M_ck = state.M_cross[a].get(k, 0.0) + state.M_cross[b].get(k, 0.0)
            state.M_cross[c][k] = M_ck
            state.M_cross[k][c] = M_ck

            state.b_cross[c][k] = (state.b_cross[a].get(k, zero_d)
                                   + state.b_cross[b].get(k, zero_d))
            state.b_cross[k][c] = (state.b_cross[k].get(a, zero_d)
                                   + state.b_cross[k].get(b, zero_d))

            state.Across[c][k] = (state.Across[a].get(k, zero_d)
                                  + state.Across[b].get(k, zero_d))
            state.Across[k][c] = (state.Across[k].get(a, zero_d)
                                  + state.Across[k].get(b, zero_d))

            state.M_cross[k].pop(a, None)
            state.M_cross[k].pop(b, None)
            state.b_cross[k].pop(a, None)
            state.b_cross[k].pop(b, None)
            state.Across[k].pop(a, None)
            state.Across[k].pop(b, None)

        state.M_local[c]  = M_c
        state.b_local[c]  = b_local_c
        state.b_global[c] = b_global_c
        state.rep[c]      = rep_c
        state.Aw[c]       = Aw_c
        state.Aloc[c]     = Aloc_c
        state.nodes[c]    = state.nodes[a] | state.nodes[b]

        # Remove merged nodes a and b from all state dicts
        for table in (state.M_local, state.b_local, state.b_global, state.rep,
                      state.M_cross, state.b_cross, state.nodes,
                      state.Aw, state.Aloc, state.Across):
            del table[a]
            del table[b]
        state.active.discard(a)
        state.active.discard(b)
        state.active.add(c)

        F_total     += delta
        merge_count += 1

        if verbose and merge_count % 25 == 0:
            total_active = sum(len(s.active) for s in states)
            print(f"  {merge_count} merges (layer {layer_idx}), "
                  f"{total_active} active clusters (all layers), "
                  f"F_total={F_total:.4f}")

        # Push new merge candidates involving the new cluster c
        push_candidates_for(layer_idx, c)

    # Collect final clusters and representatives for each layer
    results = []
    for state in states:
        active_list = list(state.active)
        clusters    = [state.nodes[k] for k in active_list]
        reps        = [state.rep[k]   for k in active_list]
        results.append((clusters, reps))

    return results, F_total

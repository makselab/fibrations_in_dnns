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
    M_local  = {i: 0.0                        for i in range(num_nodes)}
    b_local  = {i: np.zeros(feature_dim)      for i in range(num_nodes)}
    b_global = {i: weighted_features[i].copy() for i in range(num_nodes)}
    rep      = {i: w[i].copy()                for i in range(num_nodes)}
    M_cross  = {i: {}                         for i in range(num_nodes)}
    b_cross  = {i: {}                         for i in range(num_nodes)}

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


def _F_diag(mass_k, rep_k, b_global_k, A):
    """Diagonal term of the objective F for a single cluster k."""
    return float(mass_k * (rep_k @ A @ rep_k) - 2 * (rep_k @ A @ b_global_k))


def merge_delta(a, b, M_local, b_local, b_global, rep, M_cross, b_cross, A):
    """
    Compute the change in F (delta) from merging clusters a and b,
    and return the state of the resulting merged cluster c.
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

    if M_c > 1e-15:
        rep_c = b_local_c / M_c
    else:
        rep_c = 0.5 * (rep[a] + rep[b])

    neighbors = (set(M_cross[a]) | set(M_cross[b])) - {a, b}

    # Cost before merge: diagonal terms of a and b plus their cross-interaction
    F_before = (_F_diag(M_local[a], rep[a], b_global[a], A)
                + _F_diag(M_local[b], rep[b], b_global[b], A)
                + 2 * M_ab * (rep[a] @ A @ rep[b]))
    for k in neighbors:
        M_ak = M_cross[a].get(k, 0.0)
        M_bk = M_cross[b].get(k, 0.0)
        F_before += 2 * M_ak * (rep[a] @ A @ rep[k]) + 2 * M_bk * (rep[b] @ A @ rep[k])

    # Cost after merge: diagonal term of c plus its cross-interactions with neighbors
    F_after = _F_diag(M_c, rep_c, b_global_c, A)
    for k in neighbors:
        M_ck = M_cross[a].get(k, 0.0) + M_cross[b].get(k, 0.0)
        F_after += 2 * M_ck * (rep_c @ A @ rep[k])

    delta = F_after - F_before
    return delta, rep_c, M_c, b_local_c, b_global_c


# ------------------------------------------------------------------
# Multi-layer clustering with shared budget
# ------------------------------------------------------------------

class LayerState:
    """Holds the mutable clustering state for one layer."""
    __slots__ = ("M_local", "b_local", "b_global", "rep", "M_cross",
                 "b_cross", "nodes", "active", "new_id_gen", "A")

    def __init__(self, w, S, A):
        (self.M_local, self.b_local, self.b_global, self.rep, self.M_cross, self.b_cross, self.nodes) = build_initial_structures(w, S, A)
        self.active = set(self.nodes.keys())
        self.new_id_gen = itertools.count(max(self.active) + 1 if self.active else 0)
        self.A = A


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
            delta, *_ = merge_delta(node_a, node_b, state.M_local, state.b_local, state.b_global,
                                    state.rep, state.M_cross, state.b_cross, state.A)
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
        (delta, rep_c, M_c, b_local_c, b_global_c) = merge_delta(a, b, state.M_local, state.b_local, state.b_global,
                                   state.rep, state.M_cross, state.b_cross, state.A)

        if F_total + delta > F_max_total:
            break

        # Create merged cluster c and update its cross-connections to neighbors
        c = next(state.new_id_gen)
        neighbors = (set(state.M_cross[a]) | set(state.M_cross[b])) - {a, b}

        state.M_cross[c] = {}
        state.b_cross[c] = {}
        for k in neighbors:
            M_ck = state.M_cross[a].get(k, 0.0) + state.M_cross[b].get(k, 0.0)
            state.M_cross[c][k] = M_ck
            state.M_cross[k][c] = M_ck

            state.b_cross[c][k] = (state.b_cross[a].get(k, np.zeros_like(rep_c)) + state.b_cross[b].get(k, np.zeros_like(rep_c)))
            state.b_cross[k][c] = (state.b_cross[k].get(a, np.zeros_like(rep_c)) + state.b_cross[k].get(b, np.zeros_like(rep_c)))

            state.M_cross[k].pop(a, None)
            state.M_cross[k].pop(b, None)
            state.b_cross[k].pop(a, None)
            state.b_cross[k].pop(b, None)

        state.M_local[c]  = M_c
        state.b_local[c]  = b_local_c
        state.b_global[c] = b_global_c
        state.rep[c]      = rep_c
        state.nodes[c]    = state.nodes[a] | state.nodes[b]

        # Remove merged nodes a and b from all state dicts
        for table in (state.M_local, state.b_local, state.b_global, state.rep, state.M_cross, state.b_cross, state.nodes):
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
        clusters = [state.nodes[k] for k in active_list]
        reps     = [state.rep[k]   for k in active_list]
        results.append((clusters, reps))

    return results, F_total

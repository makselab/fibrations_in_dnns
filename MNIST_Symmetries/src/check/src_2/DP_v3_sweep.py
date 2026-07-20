# =====================================================
# MODULES

import argparse
import os
import torch
import numpy as np
import pandas as pd
from coloring import fibration_linear

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
args = parser.parse_args()
dev = torch.device("cuda:0")

# =====================================================
# Model

filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/train_dir/' + args.exp_name + '/gradients.pth'
data = torch.load(filename)
last_point = data[-1]
grads = {p: torch.stack([x[p]['grad'] for x in data]).mean(dim=0) for p in last_point.keys()}

hidden_size = last_point['fc1.weight']['param'].shape[0]
num_params = sum(v['param'].numel() for v in last_point.values())
print('Num Params:', num_params)

N_LAYERS = 3
num_thresholds_per_layer = 21
thresholds = torch.linspace(0, 1, num_thresholds_per_layer)

# =====================================================
# HELPERS

def compute_colors(layer_idx, in_colors, threshold):
    key_w = f'fc{layer_idx+1}.weight'
    key_b = f'fc{layer_idx+1}.bias'
    is_first = (layer_idx == 0)
    return fibration_linear(
        weights=last_point[key_w]['param'],
        in_clusters=in_colors,
        threshold=threshold,
        first_layer=is_first,
        bias=last_point[key_b]['param']
    )

def compute_dL(layer_idx, colors_layer):
    key_w = f'fc{layer_idx+1}.weight'
    key_b = f'fc{layer_idx+1}.bias'
    W  = last_point[key_w]['param']
    b  = last_point[key_b]['param']
    gW = grads[key_w]
    gb = grads[key_b]
    n_clusters = torch.unique(colors_layer).shape[0]
    P  = torch.zeros(n_clusters, hidden_size).scatter_(0, colors_layer.unsqueeze(0), 1).to(dev)
    sz = torch.mm(P, torch.ones(hidden_size, 1).to(dev))
    W_coll = (P @ W) / sz.view(-1, 1)
    b_coll = (P @ b) / sz.view(-1,)
    W_exp = W_coll[colors_layer]
    b_exp = b_coll[colors_layer]
    dL = ((W_exp - W) * gW).sum(dim=1) + (b_exp - b) * gb
    return dL.sum().item()

def pareto_filter(entries):
    """Keep entries not dominated in (dL_accum, params_accum)."""
    entries = sorted(entries, key=lambda x: (x[0], x[1]))
    kept = []
    min_params = float('inf')
    for e in entries:
        if e[1] < min_params:
            kept.append(e)
            min_params = e[1]
    return kept

# =====================================================
# PRECOMPUTE lower bounds (independent of dL_thr)

print('Precomputing lower bounds...')
K_min = [None] * N_LAYERS

colors_lb = compute_colors(0, None, 1.0)
K_min[0]  = torch.unique(colors_lb).shape[0]

for l in range(1, N_LAYERS):
    in_colors_min = torch.zeros(hidden_size, dtype=torch.long)
    colors_lb = compute_colors(l, in_colors_min, 1.0)
    K_min[l]  = torch.unique(colors_lb).shape[0]

min_params_from = [0] * (N_LAYERS + 1)
min_params_from[N_LAYERS] = 10 * (K_min[N_LAYERS - 1] + 1)
for l in range(N_LAYERS - 1, -1, -1):
    d_in = 784 if l == 0 else K_min[l - 1]
    min_params_from[l] = K_min[l] * (d_in + 1) + min_params_from[l + 1]

print(f'  K_min per layer    : {K_min}')
print(f'  min_params_from[l] : {min_params_from}')

# =====================================================
# PRECOMPUTE cache: all (in_colors_key, t_idx) -> (colors_key, dL, params)
# Done once, reused for every dL_thr value.

print('Precomputing cache layer 0...')
cache_l0 = {}
for i, t in enumerate(thresholds):
    colors = compute_colors(0, None, t.item())
    key    = tuple(colors.tolist())
    dL     = compute_dL(0, colors)
    n      = torch.unique(colors).shape[0]
    params = n * (784 + 1)
    cache_l0[i] = (key, dL, params)

print('Precomputing cache layer 1...')
unique_l0 = set(v[0] for v in cache_l0.values())

cache_l1 = {}
for in_key in unique_l0:
    in_colors = torch.tensor(list(in_key))
    n_in = int(in_colors.max().item()) + 1
    for i, t in enumerate(thresholds):
        colors = compute_colors(1, in_colors, t.item())
        key    = tuple(colors.tolist())
        dL     = compute_dL(1, colors)
        n_out  = torch.unique(colors).shape[0]
        params = n_out * (n_in + 1)
        cache_l1[(in_key, i)] = (key, dL, params)

print('Precomputing cache layer 2...')
unique_l1 = set(v[0] for v in cache_l1.values())

cache_l2 = {}
for in_key in unique_l1:
    in_colors = torch.tensor(list(in_key))
    n_in = int(in_colors.max().item()) + 1
    for i, t in enumerate(thresholds):
        colors = compute_colors(2, in_colors, t.item())
        key    = tuple(colors.tolist())
        dL     = compute_dL(2, colors)
        n_out  = torch.unique(colors).shape[0]
        params = n_out * (n_in + 1)
        cache_l2[(in_key, i)] = (key, dL, params)

print('Precomputation done.\n')

# =====================================================
# DP for a single dL_thr (uses cache + lower bound pruning)
# =====================================================

def run_dp(dL_thr):
    best_params = float('inf')
    best_entry  = None

    # --- Layer 0 ---
    dp = {}
    for i, t in enumerate(thresholds):
        colors_key, dL, params = cache_l0[i]
        if dL > dL_thr:
            continue
        if params + min_params_from[1] >= best_params:
            continue
        dp.setdefault(colors_key, []).append((dL, params, [(0, t.item())]))

    for k in dp:
        dp[k] = pareto_filter(dp[k])

    # --- Layer 1 ---
    dp_next = {}
    for in_key, entries in dp.items():
        for i, t in enumerate(thresholds):
            colors_key, dL, params = cache_l1[(in_key, i)]
            for (dL_acc, params_acc, path) in entries:
                new_dL = dL_acc + dL
                if new_dL > dL_thr:
                    continue
                new_params = params_acc + params
                if new_params + min_params_from[2] >= best_params:
                    continue
                dp_next.setdefault(colors_key, []).append(
                    (new_dL, new_params, path + [(1, t.item())]))

    for k in dp_next:
        dp_next[k] = pareto_filter(dp_next[k])
    dp = {k: v for k, v in dp_next.items() if v}

    # --- Layer 2 (last hidden layer) ---
    dp_next = {}
    for in_key, entries in dp.items():
        for i, t in enumerate(thresholds):
            colors_key, dL, params = cache_l2[(in_key, i)]
            for (dL_acc, params_acc, path) in entries:
                new_dL = dL_acc + dL
                if new_dL > dL_thr:
                    continue
                new_params = params_acc + params
                if new_params + min_params_from[3] >= best_params:
                    continue
                new_path   = path + [(2, t.item())]

                # Update best solution online
                n_last = max(colors_key) + 1
                total  = new_params + 10 * (n_last + 1)
                if total < best_params:
                    best_params = total
                    best_entry  = (new_dL, total, new_path)

                dp_next.setdefault(colors_key, []).append(
                    (new_dL, new_params, new_path))

    return best_entry, best_params

# =====================================================
# SWEEP dL_thr from 0 to 0.12 step 0.01
# =====================================================

dL_thresholds = np.arange(0.0, 0.13, 0.01)
results = []

for dL_thr in dL_thresholds:
    dL_thr = round(dL_thr, 10)
    best_entry, best_params = run_dp(dL_thr)

    if best_entry is None:
        print(f'dL_thr={dL_thr:.2f}  ->  no feasible solution')
        row = {'dL_thr': dL_thr, 'reduction_pars_coll': None,
               'dL_accum': None, 'thr1': None, 'thr2': None, 'thr3': None}
    else:
        dL_acc, total_params, path = best_entry
        reduction = total_params / num_params

        print(f'dL_thr={dL_thr:.2f}  ->  reduction={reduction:.4f}  dL={dL_acc:.6f}  '
              + '  '.join(f't{l+1}={t:.4f}' for l, t in path))
        row = {'dL_thr': dL_thr, 'reduction_pars_coll': reduction, 'dL_accum': dL_acc, **{f'thr{l+1}': t for l, t in path}}

    results.append(row)

df = pd.DataFrame(results)
out_path = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/dL_dp_sweep_' + args.exp_name + '_v3.csv'
df.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')

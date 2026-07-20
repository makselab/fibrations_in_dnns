# =====================================================
# MODULES

import argparse
import torch
import numpy as np
import pandas as pd
from coloring import fibration_linear

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name',   type=str,   required=True, help='Exp Name')
parser.add_argument('-dL_thr',     type=float, required=True, help='dL threshold')
parser.add_argument('-beam_size',  type=int,   default=20,    help='Beam size')
args = parser.parse_args()
dev = torch.device("cuda:0")

# =====================================================
# Model

filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/train_dir/' + args.exp_name + '/gradients.pth'
data = torch.load(filename)
last_point = data[-1]
grads = {p: torch.stack([x[p]['grad'] for x in data]).mean(dim=0) for p in last_point.keys()}

hidden_size = last_point['fc1.weight']['param'].shape[0]
num_params  = sum(v['param'].numel() for v in last_point.values())
print('Num Params:', num_params)

# =====================================================
# HELPERS

def compute_colors(layer_idx, in_colors, threshold):
    '''


    '''

    key_w    = f'fc{layer_idx+1}.weight'
    key_b    = f'fc{layer_idx+1}.bias'
    is_first = (layer_idx == 0)
    return fibration_linear(
        weights     = last_point[key_w]['param'],
        in_clusters = in_colors,
        threshold   = threshold,
        first_layer = is_first,
        bias        = last_point[key_b]['param']
    )

def compute_dL(layer_idx, colors_layer):
    '''

    '''
    
    key_w = f'fc{layer_idx+1}.weight'
    key_b = f'fc{layer_idx+1}.bias'
    W     = last_point[key_w]['param']
    b     = last_point[key_b]['param']
    gW    = grads[key_w]
    gb    = grads[key_b]
    n_clusters = colors_layer.max().item() + 1
    P      = torch.zeros(n_clusters, hidden_size).scatter_(0, colors_layer.unsqueeze(0), 1).to(dev)
    sz     = torch.mm(P, torch.ones(hidden_size, 1).to(dev))
    W_coll = (P @ W) / sz.view(-1, 1)
    b_coll = (P @ b) / sz.view(-1,)
    W_exp  = W_coll[colors_layer]
    b_exp  = b_coll[colors_layer]
    dL     = ((W_exp - W) * gW).sum(dim=1) + (b_exp - b) * gb
    return dL.sum().item()

def prune_beam(candidates, beam_size):
    """Keep beam_size best candidates sorted by (params_acc, dL_acc)."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: (x[1], x[0]))
    return candidates[:beam_size]

# =====================================================
# BEAM SEARCH
#
# beam entry: (dL_acc, params_acc, colors_key, path)
# path: list of (layer_idx, threshold_value)
# =====================================================

num_thr    = 21
thresholds = torch.linspace(0, 1, num_thr)
dL_thr     = args.dL_thr
beam_size  = args.beam_size

# --- Layer 0 ---
print('Layer 0...')
beam = []
for i, t in enumerate(thresholds):
    colors = compute_colors(0, None, t.item())
    dL     = compute_dL(0, colors)
    if dL > dL_thr:
        continue
    n_out  = colors.max().item() + 1
    params = n_out * (784 + 1)
    beam.append((dL, params, tuple(colors.tolist()), [(0, t.item())]))
beam = prune_beam(beam, beam_size)
print(f'  beam size: {len(beam)}')

if not beam:
    print('No feasible solution at layer 0.')
    exit()

# --- Layer 1 ---
print('Layer 1...')
candidates = []
seen = {}  # (in_key, t_idx) -> (colors, dL, params) to avoid recomputing same coloring
for (dL_acc, params_acc, in_key, path) in beam:
    in_colors = torch.tensor(list(in_key))
    n_in      = in_colors.max().item() + 1
    for i, t in enumerate(thresholds):
        cache_key = (in_key, i)
        if cache_key not in seen:
            colors = compute_colors(1, in_colors, t.item())
            dL     = compute_dL(1, colors)
            n_out  = colors.max().item() + 1
            params = n_out * (n_in + 1)  # FUNCTION -- NUMS OF PARAMS
            seen[cache_key] = (tuple(colors.tolist()), dL, params)
        colors_key, dL, params = seen[cache_key]
        new_dL = dL_acc + dL
        if new_dL > dL_thr:
            continue
        candidates.append((new_dL, params_acc + params, colors_key, path + [(1, t.item())]))
beam = prune_beam(candidates, beam_size)
print(f'  colorings computed: {len(seen)}  |  beam size: {len(beam)}')

if not beam:
    print('No feasible solution at layer 1.')
    exit()

# --- Layer 2 ---
print('Layer 2...')
candidates = []
seen = {}
for (dL_acc, params_acc, in_key, path) in beam:
    in_colors = torch.tensor(list(in_key))
    n_in      = in_colors.max().item() + 1
    for i, t in enumerate(thresholds):
        cache_key = (in_key, i)
        if cache_key not in seen:
            colors = compute_colors(2, in_colors, t.item())
            dL     = compute_dL(2, colors)
            n_out  = colors.max().item() + 1
            params = n_out * (n_in + 1)  # num of parameters of network
            seen[cache_key] = (tuple(colors.tolist()), dL, params)
        colors_key, dL, params = seen[cache_key]
        new_dL = dL_acc + dL
        if new_dL > dL_thr:
            continue
        candidates.append((new_dL, params_acc + params, colors_key, path + [(2, t.item())]))
beam = prune_beam(candidates, beam_size)
print(f'  colorings computed: {len(seen)}  |  beam size: {len(beam)}')

if not beam:
    print('No feasible solution at layer 2.')
    exit()

# --- Output layer (fc4, not compressed) ---
best_params = float('inf')
best_entry  = None
for (dL_acc, params_acc, colors_key, path) in beam:
    n_last     = max(colors_key) + 1
    params_fc4 = 10 * (n_last + 1)
    total      = params_acc + params_fc4
    if total < best_params:
        best_params = total
        best_entry  = (dL_acc, total, path)

# =====================================================
# OUTPUT
# =====================================================

if best_entry is None:
    print('No feasible solution.')
else:
    dL_acc, total_params, path = best_entry
    reduction = total_params / num_params
    print(f'\nResult:')
    print(f'  reduction = {reduction:.4f}')
    print(f'  dL_accum  = {dL_acc:.6f}')
    for l, t in path:
        print(f'  thr{l+1}     = {t:.4f}')
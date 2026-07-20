# =====================================================
# MODULES

import argparse
import torch
import pandas as pd
from model import MLP
from symmetries.coloring import fibration_linear, covering
from symmetries.collapse import collapse_linear

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHtrain', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHres', type=str, required=True, help='Exp Name')
parser.add_argument('-num_distance_thr', type=int,   default=21,    help='Number of thresholds')
parser.add_argument('-max_dL', type=float,   default=0.3,    help='Max coordinate descent iterations')
parser.add_argument('-num_distance_thr_loss', type=int,   default=31,    help='Number of thresholds')
parser.add_argument('-opfiber_threshold', type=float,   default=1.0,    help='Max coordinate descent iterations')

args = parser.parse_args()
dev = torch.device("cuda:0")

# =====================================================
# Model

filename =  args.PATHtrain + args.exp_name + '/gradients_training.pth'
data = torch.load(filename)
last_point = data[-1]
grads = {p: torch.stack([x[p]['grad'] for x in data]).mean(dim=0) for p in last_point.keys()}

# =====================================================
# Model

mlp  = MLP(input_size=784, hidden_sizes=[500, 500, 500], num_classes=10).to(dev)
dims = mlp.dims

for name, module in mlp.named_modules():
    if hasattr(module, 'weight') and f'{name}.weight' in last_point:
        module.weight.data = last_point[f'{name}.weight']['param']
        module.bias.data   = last_point[f'{name}.bias']['param']

num_params = sum(p.numel() for p in mlp.parameters())
print('Num Params:', num_params)

thresholds        = torch.linspace(0, 1, args.num_distance_thr)
dL_thrs           = torch.linspace(0, args.max_dL, args.num_distance_thr_loss)
clustering_method = {'name': 'agg_clustering', 'cfg': {'linkage': 'average'}}

# =====================================================
# PRECOMPUTE OPFIBERS

opf_thr = torch.full((mlp.num_layers,), args.opfiber_threshold)
mlp.opfibration_coloring(clustering_method, distance_thrs=opf_thr)

# =====================================================
# HELPERS

def compute_dL_and_params(idx_layer, in_colors, distance_thr):

    fib_colors = fibration_linear(weights=mlp.layers[idx_layer].weight.data, 
                            bias=mlp.layers[idx_layer].bias.data, 
                            in_colors=in_colors, 
                            clustering_method=clustering_method, 
                            distance_thr=distance_thr)

    colors = covering(fib_colors, mlp.symmetries['opfibration'][idx_layer])

    key    = tuple(colors.tolist())

    mlp.layers[idx_layer].in_colors = in_colors
    mlp.layers[idx_layer].out_colors = colors

    coll_layer, dW, db = collapse_linear(mlp.layers[idx_layer])
    params_layer = sum(p.numel() for p in coll_layer.parameters())
    dL_layer = (dW * grads[f'layers.{idx_layer}.weight']).sum().item() + (db * grads[f'layers.{idx_layer}.bias']).sum().item()

    return key, dL_layer, params_layer


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
# PRECOMPUTE all colors and dLs (independent of dL_thr)
#
# cache_l0[((), t_idx)] = (colors_key, dL, params)
# cache_l1[(in_colors_key, t_idx)] = (colors_key, dL, params)
# cache_l2[(in_colors_key, t_idx)] = (colors_key, dL, params)
# =====================================================

cache = {}
unique_states = {'l-1':[()]}

for idx_layer in range(mlp.num_layers):
    print(f'Precomputing Layer {idx_layer}')
    cache[f'l{idx_layer}'] = {}

    for in_key in unique_states[f'l{idx_layer-1}']:
        in_colors = torch.tensor(list(in_key)) if in_key else torch.arange(mlp.dims[0])
        
        for i, t in enumerate(thresholds):
            cache[f'l{idx_layer}'][(in_key, i)] = compute_dL_and_params(idx_layer, in_colors, t.item())  

    unique_states[f'l{idx_layer}'] = list(set(v[0] for v in cache[f'l{idx_layer}'].values()))

print('Precomputation done.\n')

# =====================================================
# DP for a single dL_thr (uses precomputed cache)
# =====================================================

def run_dp(dL_thr):

    dp = {():[(0,0,[])]}
    best_params = float('inf')
    best_entry  = None

    for idx_layer in range(mlp.num_layers):
        dp_next = {}
        layer_cache = cache[f'l{idx_layer}']

        for in_key, entries in dp.items():
            for i, t in enumerate(thresholds):
                colors_key, dL, params = layer_cache[(in_key, i)]
                for (dL_acc, params_acc, path) in entries:
                    new_dL = dL_acc + dL
                    if new_dL <= dL_thr:
                        candidate = (new_dL, params_acc + params, path + [(idx_layer, t.item())])
                        dp_next.setdefault(colors_key, []).append(candidate)

        dp = {k: pareto_filter(v) for k, v in dp_next.items() if v}

    # --- Output layer (fc4, not compressed) ---

    for colors_key, entries in dp.items():
        params_fc4 = mlp.dims[-1] * (len(set(colors_key)) + 1)
        for (dL_acc, params_acc, path) in entries:
            total = params_acc + params_fc4
            if total < best_params:
                best_params = total
                best_entry  = (dL_acc, total, path)

    return best_entry

# =====================================================
# SWEEP dL_thr from 0 to 0.12 step 0.01
# =====================================================

results = []

for dL_thr in dL_thrs:
    dL_thr = round(dL_thr.item(), 10)
    best = run_dp(dL_thr)

    if best is None:
        print(f'dL_thr={dL_thr:.2f}  ->  No feasible solution')
    else:
        dL_acc, total_params, path = best
        reduction = total_params / num_params
        thrs = {f'thr_fib_{l}': t for l, t in path}
        print(f'dL_thr={dL_thr:.2f}  ->  reduction={reduction:.4f}  dL={dL_acc:.6f}  ' + '  '.join(f't{l+1}={t:.4f}' for l, t in path))
        row = {'dL_thr': dL_thr, 'reduction_pars_coll': reduction, 'dL_cov': dL_acc, **thrs}
        results.append(row)

df = pd.DataFrame(results)
out_path = args.PATHres + args.exp_name + f'/optimal_curves/opfib_{args.opfiber_threshold}.csv'
df.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')
# =====================================================
# MODULES

import argparse
import itertools

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
parser.add_argument('-num_distance_thr', type=int,   default=11,    help='Number of thresholds')
parser.add_argument('-max_dL', type=float,   default=0.3,    help='Max coordinate descent iterations')
parser.add_argument('-num_distance_thr_loss', type=int,   default=31,    help='Number of thresholds')

args = parser.parse_args()
dev = torch.device("cuda:0")

# =====================================================
# Model

filename =  args.PATHtrain + args.exp_name + '/gradients_training.pth'
data = torch.load(filename)
last_point = data[-1]
grads = {p: torch.stack([x[p]['grad'] for x in data]).mean(dim=0) for p in last_point.keys()}

# hidden_size = last_point['fc1.weight']['param'].shape[0]

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

thresholds        = torch.linspace(0.4, 1, args.num_distance_thr)
dL_thrs           = torch.linspace(0, args.max_dL, args.num_distance_thr_loss)
clustering_method = {'name': 'agg_clustering', 'cfg': {'linkage': 'average'}}

# ======================================================
# COLORING & COLLAPSE
data = []

for idx_th, thrs in enumerate(itertools.product(thresholds, repeat=6)):
	print(idx_th,'/1771560')

	mlp.covering_coloring(clustering_method, fib_thrs=torch.tensor(thrs)[:3], op_thrs=torch.tensor(thrs)[3:])
	dWs, _, total_params = mlp.compute_dWs_and_params(mlp.symmetries['covering'])
	dL = 0.0

	for idx_layer in range(mlp.num_layers):
		dL += (dWs[f'layers.{idx_layer}.weight'] * grads[f'layers.{idx_layer}.weight']).sum().item()

	row = {
	    **{f"fib_{i}": thrs[i].item() for i in range(mlp.num_layers)},
	    **{f"opf_{i}": thrs[i+3].item() for i in range(mlp.num_layers)},
	    'dL': dL,
		'reduction_pars_coll': total_params/num_params}

	data.append(row)

df = pd.DataFrame(data)
results_filename = args.PATHres + args.exp_name + '/Full_Grid_dL_vs_thr.csv'
df.to_csv(results_filename, index=False)
# =====================================================
# MODULES

import argparse
import os

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss
from model import MLP
from symmetries.dL_matrix import compute_dL, compute_dL_v2
from symmetries.loss_collapse import collapse_linear as loss_collapse_linear
from symmetries.loss_collapse import collapse_linear_v2 as loss_collapse_linear_v2

import pandas as pd

DL_FNS = {'v1': compute_dL, 'v2': compute_dL_v2}
COLLAPSE_FNS = {'v1': loss_collapse_linear, 'v2': loss_collapse_linear_v2}

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name',           type=str,   required=True,            help='Exp Name')
parser.add_argument('-PATHtrain',          type=str,   required=True,            help='Training directory')
parser.add_argument('-PATHresults',        type=str,   required=True,            help='Results directory')
parser.add_argument('-PATHdata',           type=str,   required=True,            help='Dataset directory')
parser.add_argument('-epoch',              type=int,   required=True,            help='Epoch')
parser.add_argument('-distance_threshold', type=float, required=True, nargs='+', help='List of distance thresholds')
parser.add_argument('-version',            type=str,   default='v2',              help='dL and collapse version: v1 or v2')

args = parser.parse_args()

dL_fn      = DL_FNS[args.version]
collapse_fn = COLLAPSE_FNS[args.version]

dev = torch.device("cuda:0")

# =====================================================
# Load model

model_filename = args.PATHtrain + args.exp_name + '/checkpoints/model_batch_' + str(args.epoch) + '.pth'
net = torch.load(model_filename)
net.to(dev)
num_params = sum(p.numel() for p in net.parameters())
num_nodes  = sum(net.dims[1:-1])
print('Num Params:', num_params)
print('Num Nodes:', num_nodes)

# =====================================================
# Dataset (full test set for S accumulation)

test_data = MNIST(root=args.PATHdata, train=False, transform=ToTensor())
test_gen  = DataLoader(dataset=test_data, batch_size=500, shuffle=False)

criterion = CrossEntropyLoss()

# =====================================================
# Loss Coloring & Collapse

data = []

for thr in args.distance_threshold:
    print(f'distance_threshold: {thr}')

    net.loss_coloring(test_gen, criterion, thr, dL_fn=dL_fn)
    num_colors     = net.num_colors('loss')
    num_nodes_loss = sum(num_colors)

    net_loss        = net.collapse_loss_version(collapse_fn=collapse_fn)
    num_params_loss = sum(p.numel() for p in net_loss.parameters())

    name_loss   = args.exp_name + '_epoch_' + str(args.epoch) + f'_loss_{args.version}_thr_{thr}'
    loss_folder = args.PATHtrain + name_loss + '/checkpoints/'
    if not os.path.exists(loss_folder): os.makedirs(loss_folder)
    torch.save(net_loss, loss_folder + 'model_batch_0.pth')

    data.append({
        'distance_threshold':  thr,
        'num_colors_l1':       num_colors[0],
        'num_colors_l2':       num_colors[1],
        'num_colors_l3':       num_colors[2],
        'num_nodes_loss':      num_nodes_loss,
        'reduction_nodes':     num_nodes_loss / num_nodes,
        'num_params_loss':     num_params_loss,
        'reduction_pars_loss': num_params_loss / num_params,
    })

df = pd.DataFrame(data)
results_filename = args.PATHresults + args.exp_name + f'/compression_loss_results_{args.version}_{args.epoch}.csv'
df.to_csv(results_filename, index=False)

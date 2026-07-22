# =====================================================
# MODULES

import argparse
import os

import torch
import numpy as np

# =====================================================
# Load args, paths.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-epoch', type=int, required=True, help='Epoch')
parser.add_argument('-num_thrs', type=int, default=100, help='Number of thresholds')
parser.add_argument('-max_thr', type=float, default=2.0, help='Max threshold value')

args = parser.parse_args()

clustering_method = {'name': 'agg_clustering', 'cfg': {'linkage': 'average'}}

# =====================================================
# MODEL

model_filename = args.PATHtrain + args.exp_name + '/checkpoints/model_batch_' + str(args.epoch) + '.pth'
net = torch.load(model_filename)
net.eval()

hidden_size_l1 = net.dims[1]

# =====================================================
# COLORING - LAYER 1 GRID

thresholds = np.linspace(0, args.max_thr, args.num_thrs)
colors_l1 = np.zeros((args.num_thrs, 1 + hidden_size_l1))

fib_thrs = torch.zeros(net.num_layers)

for idx_thr, thr in enumerate(thresholds):
    fib_thrs[0] = thr
    net.fibration_coloring(clustering_method, fib_thrs)

    colors_l1[idx_thr, 0] = thr
    colors_l1[idx_thr, 1:] = net.symmetries['fibration'][0].numpy()

colors = {'L1': colors_l1}

# =====================================================
# SAVE

coloring_folder = args.PATHresults + args.exp_name + '/coloring/'
if not os.path.exists(coloring_folder): os.makedirs(coloring_folder)

torch.save(colors, coloring_folder + 'fibration_L1_batch_' + str(args.epoch) + '.pth')

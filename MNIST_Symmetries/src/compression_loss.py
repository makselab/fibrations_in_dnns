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

import pandas as pd

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name',   type=str,   required=True,            help='Exp Name')
parser.add_argument('-PATHtrain',  type=str,   required=True,            help='Training directory')
parser.add_argument('-PATHresults',type=str,   required=True,            help='Results directory')
parser.add_argument('-PATHdata',   type=str,   required=True,            help='Dataset directory')
parser.add_argument('-epoch',      type=int,   required=True,            help='Epoch')
parser.add_argument('-F_max',      type=float, required=True, nargs='+', help='List of F_max budget values')

args = parser.parse_args()

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
# Dataset (test set for loss_coloring)

test_data      = MNIST(root=args.PATHdata, train=False, transform=ToTensor())
test_gen       = DataLoader(dataset=test_data, batch_size=100, shuffle=False)
x_test, y_test = next(iter(test_gen))
x_test         = x_test.view(-1, 784).to(dev)
y_test         = y_test.to(dev)

criterion = CrossEntropyLoss()

# =====================================================
# Loss Coloring & Collapse

data = []

for F_max in args.F_max:
    print(f'F_max: {F_max}')

    net.loss_coloring(x_test, y_test, criterion, F_max)
    num_colors     = net.num_colors('loss')
    num_nodes_loss = sum(num_colors)

    net_loss       = net.collapse_loss_version()
    num_params_loss = sum(p.numel() for p in net_loss.parameters())

    name_loss   = args.exp_name + '_epoch_' + str(args.epoch) + f'_loss_Fmax_{F_max}'
    loss_folder = args.PATHtrain + name_loss + '/checkpoints/'
    if not os.path.exists(loss_folder): os.makedirs(loss_folder)
    torch.save(net_loss, loss_folder + 'model_batch_0.pth')

    data.append({
        'F_max':              F_max,
        'num_colors_l1':      num_colors[0],
        'num_colors_l2':      num_colors[1],
        'num_colors_l3':      num_colors[2],
        'num_nodes_loss':     num_nodes_loss,
        'reduction_nodes':    num_nodes_loss / num_nodes,
        'num_params_loss':    num_params_loss,
        'reduction_pars_loss': num_params_loss / num_params,
    })

df = pd.DataFrame(data)
results_filename = args.PATHresults + args.exp_name + f'/compression_loss_results_{args.epoch}.csv'
df.to_csv(results_filename, index=False)

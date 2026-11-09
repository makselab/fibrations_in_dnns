import argparse
from itertools import product

import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name',    type=str,   required=True)
parser.add_argument('-PATHtrain',   type=str,   required=True)
parser.add_argument('-PATHresults', type=str,   required=True)
parser.add_argument('-PATHdata',    type=str,   required=True)
parser.add_argument('-epoch',       type=int,   required=True)
parser.add_argument('-thr_min',     type=float, default=0.0)
parser.add_argument('-thr_max',     type=float, default=2.0)
parser.add_argument('-n_vals',      type=int,   default=20)
parser.add_argument('-opfib_thr',   type=float, default=1.0)
args = parser.parse_args()

dev = torch.device("cuda:0")

model_filename = args.PATHtrain + args.exp_name + '/checkpoints/model_batch_' + str(args.epoch) + '.pth'
net = torch.load(model_filename)
net.to(dev)
num_params = sum(p.numel() for p in net.parameters())
num_nodes  = sum(net.dims[1:-1])
print('Num Params:', num_params)
print('Num Nodes:', num_nodes)

test_data = MNIST(root=args.PATHdata, train=False, transform=ToTensor())
test_gen  = DataLoader(dataset=test_data, batch_size=500, shuffle=False)
criterion = CrossEntropyLoss()

clustering_method = {'name': 'agg_clustering', 'cfg': {'linkage': 'average'}}
opfib_thrs = torch.Tensor([args.opfib_thr] * net.num_layers)

thr_values = np.linspace(args.thr_min, args.thr_max, args.n_vals + 1)[1:]  # exclude 0

def evaluate(model):
    model.eval()
    correct    = 0
    total      = 0
    loss_total = 0.0
    with torch.no_grad():
        for images, labels in test_gen:
            images = images.view(-1, 784).to(dev)
            labels = labels.to(dev)
            _, out = model(images)
            _, predicted = torch.max(out, 1)
            loss_total += criterion(out, labels).item()
            correct    += (predicted == labels).sum()
            total      += labels.size(0)
    acc  = (100 * correct / (total + 1)).item()
    loss = loss_total / len(test_gen)
    return acc, loss

combos = list(product(thr_values, repeat=3))
n_total = len(combos)
print(f'Sweeping {n_total} threshold combinations...')

data = []
for idx, (thr_0, thr_1, thr_2) in enumerate(combos):
    if idx % 500 == 0:
        print(f'  {idx}/{n_total}')

    fib_thrs = torch.Tensor([thr_0, thr_1, thr_2])
    net.covering_coloring(clustering_method, fib_thrs, opfib_thrs)
    num_colors     = net.num_colors('covering')
    num_nodes_coll = sum(num_colors)

    net_coll, _ = net.collapse_version('covering')
    net_coll.to(dev)
    num_params_coll = sum(p.numel() for p in net_coll.parameters())

    acc, loss = evaluate(net_coll)

    data.append({
        'thr_fib_0':           thr_0,
        'thr_fib_1':           thr_1,
        'thr_fib_2':           thr_2,
        'num_colors_l1':       num_colors[0],
        'num_colors_l2':       num_colors[1],
        'num_colors_l3':       num_colors[2],
        'num_nodes_coll':      num_nodes_coll,
        'reduction_nodes':     num_nodes_coll / num_nodes,
        'num_params_coll':     num_params_coll,
        'reduction_pars_coll': num_params_coll / num_params,
        'acc':                 acc,
        'loss':                loss,
    })

df = pd.DataFrame(data)
results_filename = args.PATHresults + args.exp_name + f'/sensibility_analysis_{args.epoch}.csv'
df.to_csv(results_filename, index=False)
print(f'Saved to {results_filename}')
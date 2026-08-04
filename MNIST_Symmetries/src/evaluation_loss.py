# =====================================================
# MODULES

import argparse

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import CrossEntropyLoss

import pandas as pd

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name',    type=str,   required=True,            help='Exp Name')
parser.add_argument('-PATHtrain',   type=str,   required=True,            help='Training directory')
parser.add_argument('-PATHresults', type=str,   required=True,            help='Results directory')
parser.add_argument('-PATHdata',    type=str,   required=True,            help='Dataset directory')
parser.add_argument('-epoch',       type=int,   required=True,            help='Epoch')
parser.add_argument('-F_max',       type=float, required=True, nargs='+', help='List of F_max budget values')

args = parser.parse_args()

dev = torch.device("cuda:0")

# =====================================================
# Dataset

test_data = MNIST(root=args.PATHdata, train=False, transform=ToTensor())
test_gen  = DataLoader(dataset=test_data, batch_size=100, shuffle=False)

loss_function = CrossEntropyLoss()

# =====================================================
# Evaluation

def evaluate(net):
    net.eval()
    correct    = 0
    total      = 0
    loss_total = 0
    with torch.no_grad():
        for images, labels in test_gen:
            images = images.view(-1, 784).to(dev)
            labels = labels.to(dev)
            _, out = net(images)
            _, predicted = torch.max(out, 1)
            loss_total += loss_function(out, labels).item()
            correct    += (predicted == labels).sum()
            total      += labels.size(0)
    acc  = (100 * correct / (total + 1)).item()
    loss = loss_total / len(test_gen)
    return acc, loss

data = []

for F_max in args.F_max:
    print(f'F_max: {F_max}')

    name_loss   = args.exp_name + '_epoch_' + str(args.epoch) + f'_loss_Fmax_{F_max}'
    loss_folder = args.PATHtrain + name_loss + '/checkpoints/'
    net_loss    = torch.load(loss_folder + 'model_batch_0.pth', weights_only=False)
    net_loss.to(dev)

    acc_loss, loss_loss = evaluate(net_loss)

    data.append({
        'F_max':     F_max,
        'acc_loss':  acc_loss,
        'loss_loss': loss_loss,
    })

df = pd.DataFrame(data)
results_filename = args.PATHresults + args.exp_name + f'/Evaluation_LossModels_epoch_{args.epoch}.csv'
df.to_csv(results_filename, index=False)

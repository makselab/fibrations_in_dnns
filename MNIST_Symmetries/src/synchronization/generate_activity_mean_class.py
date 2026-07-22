# =====================================================
# MODULES

import argparse
import os

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-PATHdata', type=str, default=True, help='Dataset directory')
parser.add_argument('-epoch', type=int, required=True, help='Epoch')

args = parser.parse_args()

dev = torch.device("cuda:0")

# =====================================================
# DATASET

batch_size = 100
test_data = MNIST(root=args.PATHdata, train=False, transform=ToTensor())
gen = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# =====================================================
# MODEL

model_filename = args.PATHtrain + args.exp_name + '/checkpoints/model_batch_' + str(args.epoch) + '.pth'
net = torch.load(model_filename)
net.to(dev)
net.eval()

input_size = net.dims[0]

results_folder = args.PATHresults + args.exp_name + '/synchronization/'
if not os.path.exists(results_folder): os.makedirs(results_folder)

# =====================================================
# COLLECT FULL TEST SET

all_images = []
all_labels = []

for images, labels in gen:
    all_images.append(images.view(-1, input_size))
    all_labels.append(labels)

all_images = torch.cat(all_images)
all_labels = torch.cat(all_labels)

# =====================================================
# ACTIVITY - MEAN CLASS INPUTS

num_classes = len(torch.unique(all_labels))
mean_inputs = []

for cl in range(num_classes):
    idx = (all_labels == cl)
    mean_img = all_images[idx].mean(dim=0, keepdim=True)
    mean_inputs.append(mean_img)

mean_inputs = torch.cat(mean_inputs).to(dev)
mean_activations, _ = net(mean_inputs)

torch.save(mean_activations, results_folder + 'activity_mean_class_batch_' + str(args.epoch) + '.pth')

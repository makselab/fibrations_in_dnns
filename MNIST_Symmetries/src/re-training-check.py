# =====================================================
# MODULES

import os
import json
import argparse
import re

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from model import MLP


# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
parser.add_argument('-PATHcfg', type=str, required=True, help='Config directory')

args = parser.parse_args()

exp_folder = args.PATHtrain + args.exp_name

if not os.path.exists(exp_folder): 
  os.makedirs(exp_folder + '/checkpoints')

  with open(args.PATHcfg + args.exp_name + '.json', 'r') as f:
      cfg = json.load(f)
      net = None
else:
  checkpoints_files = os.listdir(exp_folder + '/checkpoints')
  checkpoint_filename = max(checkpoints_files, key=lambda x: int(re.search(r'model_batch_(\d+)\.pth', x).group(1)))
  net = torch.load(exp_folder + '/checkpoints/' + checkpoint_filename, weights_only=False)

dataPATH = "/home/ovelarde/snap/snapd-desktop-integration/253/Documents/MNIST_Symmetries/"

dev = torch.device("cuda:0")

# =====================================================
# DATASET

if net is None:
  input_size = cfg['data']['input_size']
  num_classes = cfg['data']['num_classes'] 
  batch_size = cfg['training']['batch_size']
  hidden_sizes = cfg['model']['layers']
  net = MLP(input_size, hidden_sizes, num_classes)
  lr = cfg['training']['lr']
else:
  input_size = net.fc1.weight.shape[1]
  num_classes = net.fc4.weight.shape[0]
  lr = 0.001
  batch_size = 100

train_data = MNIST(root = dataPATH, train = True, transform = ToTensor(), download=True)
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor(), download=True)

train_gen = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_gen = DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)

# =====================================================
# MODEL

net.to(dev)
loss_function = CrossEntropyLoss()

# =====================================================
# TRAINING 
optimizer = Adam(net.parameters(), lr=lr)

# =====================================================
# TRAINING

acc_values = []

for i ,(images,labels) in enumerate(train_gen):
  images = images.view(-1,input_size).to(dev)
  labels = labels.to(dev)

  torch.save(net, exp_folder + '/checkpoints/model_batch_' + str(i) + '.pth')

  # =====================================================
  # TRAINING

  net.train()

  optimizer.zero_grad()
  out_1, out_2 = net(images)
  loss = loss_function(out_2, labels)
  loss.backward()
  optimizer.step()

  # =====================================================
  # EVALUATION

  net.eval()

  correct = 0
  total = 0

  for images,labels in test_gen:
    images = images.view(-1,input_size).to(dev)
    labels = labels.to(dev)

    out_1, out_2 = net(images)
    _, predicted = torch.max(out_2,1)
    correct += (predicted == labels).sum()
    total += labels.size(0)

  acc = (100*correct)/(total+1)
  acc = acc.cpu().item()
  acc_values.append(acc)

  print(i, acc)

# =====================================================
# SAVE PERFORMANCE
torch.save(torch.tensor(acc_values), exp_folder + '/accuracy.pth')
# =====================================================
# MODULES

import os
import json
import argparse

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch import no_grad

from model import MLP

# =====================================================
# Load args, paths, device.

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
parser.add_argument('-PATHcfg', type=str, required=True, help='Config directory')

args = parser.parse_args()

with open(args.PATHcfg + args.exp_name + '.json', 'r') as f:
    cfg = json.load(f)

dataPATH = cfg['data']['dataset_path']

exp_folder = args.PATHtrain + args.exp_name
if not os.path.exists(exp_folder): 
  os.makedirs(exp_folder + '/checkpoints')

dev = torch.device("cuda:0")

# =====================================================
# DATASET

input_size = cfg['data']['input_size']
num_classes = cfg['data']['num_classes'] 
batch_size = cfg['training']['batch_size']

train_data = MNIST(root = dataPATH, train = True, transform = ToTensor(), download=True)
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor(), download=True)

train_gen = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_gen = DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)

# =====================================================
# MODEL

hidden_sizes = cfg['model']['layers']
net = MLP(input_size, hidden_sizes, num_classes)
net.to(dev)
loss_function = CrossEntropyLoss()

# =====================================================
# TRAINING 
lr = cfg['training']['lr']
optimizer = Adam(net.parameters(), lr=lr)

loss_history_test = []
loss_history_train = []
acc_values = []
all_grads_training = []

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

  grad_snapshot = {
    name: {
        'param': param.data.clone(),
        'grad': param.grad.clone()}
    for name, param in net.named_parameters() if param.grad is not None}

  all_grads_training.append(grad_snapshot)

  optimizer.step()

  loss_history_train.append(loss)

  # =====================================================
  # EVALUATION

  net.eval()

  correct = 0
  total = 0
  loss_test = 0

  with no_grad():
    for images,labels in test_gen:
      images = images.view(-1,input_size).to(dev)
      labels = labels.to(dev)

      out_1, out_2 = net(images)
      _, predicted = torch.max(out_2,1)
      correct += (predicted == labels).sum()
      total += labels.size(0)

      loss_test_batch = loss_function(out_2, labels)
      loss_test += loss_test_batch

  loss_test = loss_test/len(test_gen)
  acc = (100*correct)/(total+1)
  acc = acc.cpu().item()
  acc_values.append(acc)
  loss_history_test.append(loss_test)

  print(i, '- Train Loss: ', loss.item(), 'Test Loss: ', loss_test.item(), 'Acc: ', acc)

print('-------- Last evaluation ----------------------')

# =====================================================
# LAST EVALUATION

net.eval()

correct = 0
total = 0
loss_test = 0

all_grads_test = [] 
for images,labels in test_gen:
  optimizer.zero_grad()
  images = images.view(-1,input_size).to(dev)
  labels = labels.to(dev)

  out_1, out_2 = net(images)
  _, predicted = torch.max(out_2,1)
  correct += (predicted == labels).sum()
  total += labels.size(0)

  loss_test_batch = loss_function(out_2, labels)
  loss_test_batch.backward()
  loss_test += loss_test_batch

  grad_snapshot_test = {
    name: {
      'param': param.data.clone(),
      'grad': param.grad.clone()}
    for name, param in net.named_parameters() if param.grad is not None}

  all_grads_test.append(grad_snapshot_test)

loss_test = loss_test/len(test_gen)
print('Test Loss: ', loss_test.item())

acc = (100*correct)/(total+1)
acc = acc.cpu().item()

print(i, acc)

# =====================================================
# SAVE PERFORMANCE
torch.save(torch.tensor(acc_values), exp_folder + '/accuracy.pth')
torch.save(torch.tensor(loss_history_train), exp_folder + '/loss_train.pth')
torch.save(torch.tensor(loss_history_test), exp_folder + '/loss_test.pth')
torch.save(all_grads_training, exp_folder + '/gradients_training.pth')
torch.save(all_grads_test, exp_folder + '/gradients_test.pth')
# =====================================================
# MODULES

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model import MLP
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

# =====================================================
dev = torch.device("cuda:0")
dataPATH = '/media/osvaldo/Seagate Basic/'
resultsPATH = '/media/osvaldo/OMV5TB/MNIST_Symmetries/'

# =====================================================
# DATASET

input_size = 784 # img_size = (28,28)
num_classes = 10 
batch_size = 100 

train_data = MNIST(root = dataPATH, train = True, transform = ToTensor(), download=True)
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor(), download=True)

train_gen = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_gen = DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)

# =====================================================
# MODEL

hidden_size = 500 #100
net = MLP(input_size, [hidden_size,hidden_size,hidden_size], num_classes)

net.to(dev)
loss_function = CrossEntropyLoss()

# =====================================================
# TRAINING 
lr = 1e-3
optimizer = Adam(net.parameters(), lr=lr)

# =====================================================
# TRAINING

acc_values = []

for i ,(images,labels) in enumerate(train_gen):
  images = images.view(-1,input_size).to(dev)
  labels = labels.to(dev)

  # SAVE WEIGHTS
  torch.save(net.state_dict(), resultsPATH + 'training/weights_batch_' + str(i) + '.pth')

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
  acc_values.append(acc)

  print(i, acc)

# =====================================================
# SAVE PERFORMANCE
torch.save(torch.tensor(acc_values), resultsPATH + 'training/accuracy.pth')

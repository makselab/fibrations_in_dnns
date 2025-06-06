# =====================================================
# MODULES

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model import MLP
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

train_data = MNIST(root = dataPATH, train = True, transform = ToTensor())
test_data = MNIST(root = dataPATH, train = False, transform = ToTensor())
gen = DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)
# gen = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

# =====================================================
# MODEL
idx_batch = 599
hidden_size = 500 #100
net = MLP(input_size, [hidden_size,hidden_size,hidden_size], num_classes)
weights = torch.load(resultsPATH + 'training/weights_batch_' + str(idx_batch) + '.pth')
net.load_state_dict(weights)
net.to(dev)

# =====================================================

# EVALUATION
net.eval()

activities = [[],[],[]]
list_labels = []
list_predictions = []

for images,labels in gen:
  images = images.view(-1,input_size).to(dev)
  activations, out_2 = net(images)
  _, predicted = torch.max(out_2,1)

  for ii in range(3):
    activities[ii].append(activations[ii])

  list_labels.append(labels)
  list_predictions.append(predicted.cpu())

tensor_act = [torch.cat(act, dim=0) for act in activities]
list_labels = torch.cat(list_labels)
list_predictions = torch.cat(list_predictions)

torch.save({'activity':tensor_act, 'labels': list_labels, 'prediction': list_predictions}, resultsPATH + 'activity_batch_' + str(idx_batch) + '.pth')

# =====================================================

# RANDOM IMAGES

images = torch.rand(200, 784).to(dev)
activations, _ = net(images)

torch.save(activations, resultsPATH + 'activity_random_input_batch_' + str(idx_batch) + '.pth')

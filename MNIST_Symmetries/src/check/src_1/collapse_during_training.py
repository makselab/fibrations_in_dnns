# =====================================================
# MODULES

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import MLP

# =================================================================
# PARAMETERS - VARIABLES.

dev = torch.device("cuda:0")
dataPATH = '/media/osvaldo/Seagate Basic/'
resultsPATH = '/media/osvaldo/OMV5TB/MNIST_Symmetries/'

# =====================================================
# DATASET

input_size = 784 # img_size = (28,28)
num_classes = 10 
batch_size = 100 

test_data = MNIST(root = dataPATH, train = False, transform = ToTensor())
test_gen = DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)

# =====================================================
# MODEL

hidden_size = 500 #100
net = MLP(input_size, [hidden_size, hidden_size, hidden_size], num_classes)
net.to(dev)
num_params_original = sum(p.numel() for p in net.parameters())

# ==================================================================
idx_thresholds = [0,10,20,30,40,50,60,70,80,90,100]
num_thresholds = len(idx_thresholds)
num_batchs = 100
idx_batchs = [6*ii for ii in range(num_batchs)]
results = torch.zeros(num_thresholds, num_batchs, 6)

for idx_b_res, idx_batch in enumerate(idx_batchs):
    print('Running Batch: ', idx_batch)

    weights = torch.load(resultsPATH + 'training/weights_batch_' + str(idx_batch) + '.pth')
    colors = torch.load(resultsPATH + 'coloring/fibration_batch_' + str(idx_batch) + '.pth')
    net.load_state_dict(weights)

    for idx_t_res ,idx_thr in enumerate(idx_thresholds):
        thr = colors['L1'][idx_thr,0]
        colors_l1 = colors['L1'][idx_thr,1:].long()
        colors_l2 = colors['L2'][idx_thr,1:].long()
        colors_l3 = colors['L3'][idx_thr,1:].long()

        num_colors_l1 = torch.unique(colors_l1).shape[0]
        num_colors_l2 = torch.unique(colors_l2).shape[0]
        num_colors_l3 = torch.unique(colors_l3).shape[0]

        mtx_partition_l1 = torch.zeros(num_colors_l1, hidden_size).scatter_(0, colors_l1.unsqueeze(0), 1).to(dev)
        mtx_partition_l2 = torch.zeros(num_colors_l2, hidden_size).scatter_(0, colors_l2.unsqueeze(0), 1).to(dev)
        mtx_partition_l3 = torch.zeros(num_colors_l3, hidden_size).scatter_(0, colors_l3.unsqueeze(0), 1).to(dev)
        sizes_clusters_l1 = torch.mm(mtx_partition_l1, torch.ones(hidden_size, 1).to(dev))
        sizes_clusters_l2 = torch.mm(mtx_partition_l2, torch.ones(hidden_size, 1).to(dev))
        sizes_clusters_l3 = torch.mm(mtx_partition_l3, torch.ones(hidden_size, 1).to(dev))

        W1 = net.fc1.weight.data
        W2 = net.fc2.weight.data
        W3 = net.fc3.weight.data
        W4 = net.fc4.weight.data

        b1 = net.fc1.bias.data
        b2 = net.fc2.bias.data
        b3 = net.fc3.bias.data
        b4 = net.fc4.bias.data

        W1_coll = (mtx_partition_l1 @ W1) / sizes_clusters_l1.view(-1, 1)
        W2_coll = (mtx_partition_l2 @ W2 @ mtx_partition_l1.T) / sizes_clusters_l2.view(-1, 1)
        W3_coll = (mtx_partition_l3 @ W3 @ mtx_partition_l2.T) / sizes_clusters_l3.view(-1, 1)
        W4_coll = W4 @ mtx_partition_l3.T

        b1_coll = mtx_partition_l1 @ b1 / sizes_clusters_l1.view(-1,)
        b2_coll = mtx_partition_l2 @ b2 / sizes_clusters_l2.view(-1,)
        b3_coll = mtx_partition_l3 @ b3 / sizes_clusters_l3.view(-1,)
        b4_coll = b4

        net_coll = MLP(input_size, [num_colors_l1,num_colors_l2,num_colors_l3], num_classes)

        num_params = sum(p.numel() for p in net_coll.parameters())

        net_coll.fc1.weight.data = W1_coll
        net_coll.fc2.weight.data = W2_coll
        net_coll.fc3.weight.data = W3_coll
        net_coll.fc4.weight.data = W4_coll

        net_coll.fc1.bias.data = b1_coll
        net_coll.fc2.bias.data = b2_coll
        net_coll.fc3.bias.data = b3_coll
        net_coll.fc4.bias.data = b4_coll

        # Evaluation
        net_coll.eval()
        correct = 0
        total = 0

        for images,labels in test_gen:
            images = images.view(-1,input_size).to(dev)
            labels = labels.to(dev)

            out_1, out_2 = net_coll(images)
            _, predicted = torch.max(out_2,1)
            correct += (predicted == labels).sum()
            total += labels.size(0)

        acc = (100*correct)/(total+1)

        results[idx_t_res,idx_b_res,0]= thr.item()
        results[idx_t_res,idx_b_res,1]= num_colors_l1
        results[idx_t_res,idx_b_res,2]= num_colors_l2
        results[idx_t_res,idx_b_res,3]= num_colors_l3
        results[idx_t_res,idx_b_res,4]= num_params/num_params_original
        results[idx_t_res,idx_b_res,5]= acc.item()

torch.save({'idx_batchs':idx_batchs, 'idx_thr': idx_thresholds, 'results': results}, resultsPATH + 'collapse/during_training.pth')
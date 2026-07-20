# MODULES

import torch
from model import MLP

# =================================================================
# PARAMETERS - VARIABLES.

dev = torch.device("cuda:0")
PATH = '/media/osvaldo/OMV5TB/MNIST_Symmetries/'
# =====================================================
# MODEL
input_size = 784 # img_size = (28,28)
num_classes = 10 
hidden_size = 500 #100
net = MLP(input_size, [hidden_size,hidden_size,hidden_size], num_classes)
net.to(dev)

# ==================================================================
# COLORING
num_thresholds = 101
values_threshold= torch.linspace(0,1, num_thresholds)

for i in range(600):
    print('Batch_IDx:',i)
    weights = torch.load(PATH + 'training/weights_batch_' + str(i) + '.pth')

    net.load_state_dict(weights)

    clusters = {'L1':torch.zeros(num_thresholds,1+hidden_size),'L2':torch.zeros(num_thresholds,1+hidden_size),'L3':torch.zeros(num_thresholds,1+hidden_size)}

    for idx_th, thr in enumerate(values_threshold):
        for cc in clusters.values():
            cc[idx_th,0] = thr.item()
        
        net.fibration_coloring(threshold=thr.item(), bias=True)

        for ii in range(1,4):
            clusters['L'+str(ii)][idx_th,1:] = net.fibration_colors[ii-1]

    torch.save(clusters,PATH + 'coloring/fibration_batch_' + str(i) + '.pth')

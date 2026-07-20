import matplotlib.pyplot as plt
import torch
import numpy as np

PATH = '/media/osvaldo/OMV5TB/MNIST_Symmetries/'
data = torch.load(PATH + 'collapse/during_training.pth')#.numpy()

hidden_size = 500 #100
idx_batchs = data['idx_batchs']
idx_thr = data['idx_thr']
res = data['results'].numpy()
colors_plots = ['k','y','c','pink','b','g','r','orange','purple','olive','brown']
# ================================================

fig, axs = plt.subplots(3,1, figsize = (5,15))

# Performance - Reduction
axs[0].set_xlabel('Time Training')
axs[1].set_xlabel('Time Training')
axs[2].set_xlabel('Time Training')

axs[0].set_ylabel('Accuracy')
axs[1].set_ylabel('Reduction')
axs[2].set_ylabel('Num Fibers - L1')

axs[0].set_ylim([0,100])
axs[1].set_ylim([0,1])
axs[2].set_ylim([0,hidden_size])

for idx_t, idx_thr in enumerate(idx_thr):
	axs[0].plot(idx_batchs,res[idx_t,:,5], label = str(idx_thr/100),c=colors_plots[idx_t])
	axs[1].plot(idx_batchs,res[idx_t,:,4],c=colors_plots[idx_t])
	axs[2].plot(idx_batchs,res[idx_t,:,1], ls='-', c=colors_plots[idx_t])
	# axs[2].plot(idx_batchs,res[idx_t,:,2], ls=':',c=colors_plots[idx_t])
	# axs[2].plot(idx_batchs,res[idx_t,:,3], ls='--',c=colors_plots[idx_t])

axs[0].legend()

fig.savefig(PATH + 'plots/during_training.svg', format='svg')









# axs[1].plot(data[:,0],data[:,1], color = 'blue', label = 'L1')
# axs[1].plot(data[:,0],data[:,2], color = 'red', label = 'L2')
# axs[1].plot(data[:,0],data[:,3], color = 'green', label = 'L3')
# axs[1].set_xlabel('Threshold')
# axs[1].set_ylabel('Num Fibers')

# axs[1].legend()

# 
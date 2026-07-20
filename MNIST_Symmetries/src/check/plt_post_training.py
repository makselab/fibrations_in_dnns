import matplotlib.pyplot as plt
import torch
import numpy as np

PATH = '/media/osvaldo/OMV5TB/MNIST_Symmetries/'
data = torch.load(PATH + 'collapse/post_training.pth').numpy()

fig, axs = plt.subplots(2,1, figsize = (5,10))

# Performance - Reduction
axs0b = axs[0].twinx()

axs0b.plot(data[:,0],data[:,4], color = 'red')
axs[0].plot(data[:,0],data[:,5], color = 'blue')

axs[0].set_xlabel('Threshold')
axs0b.set_ylabel('Reduction', color = 'red')
axs0b.set_ylim([0,1])
axs[0].set_ylabel('Accuracy', color='blue')
axs[0].set_ylim([50,100])


axs[1].plot(data[:,0],data[:,1], color = 'blue', label = 'L1')
axs[1].plot(data[:,0],data[:,2], color = 'red', label = 'L2')
axs[1].plot(data[:,0],data[:,3], color = 'green', label = 'L3')
axs[1].set_xlabel('Threshold')
axs[1].set_ylabel('Num Fibers')

axs[1].legend()

fig.savefig(PATH + 'plots/post_training.svg', format='svg')
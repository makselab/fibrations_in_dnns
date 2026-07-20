import torch
import matplotlib.pyplot as plt

f0 = '/home/ovelarde/snap/snapd-desktop-integration/253/Documents/MNIST_Symmetries/train_dir/exp_01/accuracy.pth'
f1 = '/home/ovelarde/snap/snapd-desktop-integration/253/Documents/MNIST_Symmetries/train_dir/exp_01_epoch_599_coll_thr_0.8100000023841858/accuracy.pth'
f2 = '/home/ovelarde/snap/snapd-desktop-integration/253/Documents/MNIST_Symmetries/train_dir/exp_01_epoch_599_abl_thr_0.8100000023841858/accuracy.pth'
f3 = '/home/ovelarde/snap/snapd-desktop-integration/253/Documents/MNIST_Symmetries/train_dir/exp_02/accuracy.pth'

d0 = torch.load(f0)
d1 = torch.load(f1)
d2 = torch.load(f2)
d3 = torch.load(f3)

fig, axs = plt.subplots(1,1)
axs.plot(d1)
axs.plot(d2)
axs.set_ylim(0,100)
fig.savefig('retrain.svg',format='svg')


fig2, axs2 = plt.subplots(1,1)
axs2.plot(d0)
axs2.plot(d3)
axs2.set_ylim(0,100)

fig2.savefig('original.svg',format='svg')

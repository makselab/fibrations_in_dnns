import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
args = parser.parse_args()

PATH_ = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/'

filename1 = PATH_ + f'results/{args.exp_name}/Symmetries_thrs_[1.5, 1.5, 1.5, 0.8, 0.8, 0.3].csv'
filename2 = PATH_ + f'train_dir/{args.exp_name}/accuracy.pth'

df = pd.read_csv(filename1)
acc = torch.load(filename2)

size_layer = [500,500,500]

fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Performance')
# ax1.set_ylim([0.5,1])
ax1.set_xlim([0,600])
ax2 = ax1.twinx() 
ax2.set_ylabel('Fibers/Num Nodes')
ax2.set_ylim([0,1])
fig.tight_layout()  # otherwise the right y-label is slightly clipped

ax1.plot(acc, color='green')

for ii in range(3):
	ax2.plot(df['epoch'], df['num_colors_l'+str(ii)]/size_layer[ii], ls='--', label=str(ii))

ax2.legend()
fig.savefig(PATH_ + f'results/{args.exp_name}/Symmetries_vs_time_opfibers.svg',format='svg')

plt.show()
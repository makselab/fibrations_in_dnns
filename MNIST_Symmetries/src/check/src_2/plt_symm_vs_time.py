import pandas as pd
import matplotlib.pyplot as plt
import torch

filename1 = './results/Symmetries_exp_01_thr_0.8100000023841858.csv'
filename2 = './train_dir/exp_01/accuracy.pth'

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
	ax2.plot(df['epoch'], df['num_colors_l'+str(ii+1)]/size_layer[ii], ls='--', label=str(ii))

ax2.legend()
fig.savefig('symmetries_vs_time.svg',format='svg')
# =================================================================
# MODULES

import matplotlib.pyplot as plt
from pandas import read_csv
import argparse

# =================================================================
# PARAMETERS - VARIABLES.

parser = argparse.ArgumentParser()
parser.add_argument('--dataPATH', type=str)
parser.add_argument('--symmetry', type=str)
parser.add_argument('--task_idx', type=int)
args = parser.parse_args()

num_epochs = 6
max_size   = [10,20,400,150,90]
layer_size = [32,64,512,128,128]

# =================================================================

for idx_LL in range(0,5):
	df = read_csv(args.dataPATH + 'symmetries/' + args.symmetry + '/Layer_' + str(idx_LL) + '.csv', header=None)

	fig, axs = plt.subplots(1,num_epochs, figsize=(5*num_epochs,5))

	fig.suptitle('Layer ' + str(idx_LL+1) + ' - Size: ' + str(layer_size[idx_LL]))

	for ax in axs:
		ax.set_ylim([0.5,max_size[idx_LL]])
		ax.set_yscale('log')
		ax.set_xlim([0.5,layer_size[idx_LL]])
		ax.set_xscale('log')

	for ep, t in enumerate(range(args.task_idx*num_epochs, (args.task_idx+1)*num_epochs)):
		data = df.iloc[:,t]

		histogram = data.value_counts(normalize=False).reset_index(drop=True)
		axs[ep].bar(histogram.index+1, histogram.values, color='blue', edgecolor='black')

	axs[0].set_ylabel('Size')
	axs[5].set_xlabel('Id Fiber')

	fig.savefig(args.dataPATH + 'symmetries/' + args.symmetry + '/sizes/Layer_' + str(idx_LL)+ '_Task_' + str(args.task_idx) + '.svg', format='svg')
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
parser.add_argument('--layer_size', nargs='+', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--window_size', type=int)

args = parser.parse_args()

# =================================================================
# PROCESSING.
num_layers = len(args.layer_size)
fig, axs = plt.subplots(1,num_layers,figsize=(5*num_layers,5))

for idx_LL in range(num_layers):
	df = read_csv(args.dataPATH + 'symmetries/' + args.symmetry + '/Layer_' + str(idx_LL) + '.csv', header=None)

	df = df.iloc[:, args.num_epochs-1::args.num_epochs].copy()
	df.columns = range(df.columns.size)

	num_tasks = len(df.columns)

	xticks = [i * num_tasks // 5 for i in range(6)]  # [0*6, 1*6, 2*6, 3*6, 4*6, 5*6]

	num_fibers = df.nunique()
	axs[idx_LL].plot(num_fibers)

	if args.window_size>0:
		smooth_version = num_fibers.rolling(window=args.window_size, center=True).mean()
		axs[idx_LL].plot(smooth_version)

	axs[idx_LL].set_ylim([1,args.layer_size[idx_LL]])
	axs[idx_LL].set_yscale('log')
	axs[idx_LL].set_ylabel('# Fibers')
	axs[idx_LL].set_xlabel('Time')
	axs[idx_LL].set_xticks(xticks)

fig.savefig(args.dataPATH + 'symmetries/' + args.symmetry + '/num_fibers_vs_time.svg', format='svg')
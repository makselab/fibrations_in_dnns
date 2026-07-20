import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
args = parser.parse_args()

PATH_RES = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/'
datafolder = PATH_RES  + f'{args.exp_name}/optimal_curves/'
fig, axs = plt.subplots(1,1,figsize=(7,7))
axs.set_xlim([0,1.0])
axs.set_ylim([0,0.3])
axs.set_xlabel('Compressed Size')
axs.set_ylabel('dL')


for f in os.listdir(datafolder):
	filename = datafolder + f
	data = pd.read_csv(filename)
	axs.plot(data['reduction_pars_coll'],data['dL_thr'], label = 'opf_thr = ' +f.split('_')[1][:-4])

axs.legend()
fig.savefig(PATH_RES + args.exp_name + '/Optimal_dL_curves.svg',format='svg')
plt.show()
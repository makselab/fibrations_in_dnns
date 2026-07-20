import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
args = parser.parse_args()

PATH_RES = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/'
filename =  PATH_RES + f'{args.exp_name}/Ev_pareto_frontier_599.csv'
data = pd.read_csv(filename)

fig, axs = plt.subplots(2,2,figsize=(14,14))
axs[0][0].set_xlim([0,1.0])
axs[0][1].set_xlim([0,1.0])
axs[1][0].set_xlim([0,1.0])
axs[1][1].set_xlim([0,1.0])

axs[0][0].set_ylim([0,100])
axs[0][0].set_ylabel('Accuracy')
axs[1][0].set_xlabel('Compressed Size')
axs[1][0].set_ylabel('Loss')
axs[1][1].set_xlabel('Compressed Size (Nodes)')


# -----------------------------------------------------------

axs[0][0].plot(data['reduction_pars_coll'],data['acc_coll'])
axs[1][0].plot(data['reduction_pars_coll'],data['loss_coll'])

data.sort_values("reduction_nodes", inplace=True)

axs[0][1].plot(data['reduction_nodes'],data['acc_coll'])
axs[1][1].plot(data['reduction_nodes'],data['loss_coll'])

axs[0][0].legend()

fig.savefig(PATH_RES + args.exp_name + '/Optimal_Pareto_curve.svg',format='svg')

plt.show()
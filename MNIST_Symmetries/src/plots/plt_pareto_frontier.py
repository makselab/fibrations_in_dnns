import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
args = parser.parse_args()

PATH_RES = args.PATHresults
filename =  PATH_RES + f'{args.exp_name}/pareto_frontier.csv'
data = pd.read_csv(filename)

fig, axs = plt.subplots(1,1,figsize=(14,14))

axs.set_xlim(0,1)
axs.set_ylim(0,0.3)

axs.plot(data['reduction_pars_coll'],data['dL'])

fig.savefig(PATH_RES + args.exp_name + '/Optimal_Pareto_curve_dL.svg',format='svg')

plt.show()
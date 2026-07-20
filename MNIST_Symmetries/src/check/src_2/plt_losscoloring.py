import pandas as pd
import matplotlib.pyplot as plt

filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/loss_coloring_dL_vs_thr_exp_01.csv'
filename2 = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/loss_coloring_dL_vs_thr_exp_01_v1.csv'

data = pd.read_csv(filename)
data2 = pd.read_csv(filename2)

fig, axs = plt.subplots(1,1)

axs.plot(data['reduction_pars_coll'],data['acc'])
axs.plot(data2['reduction_pars_coll'],data2['acc'])

axs.set_ylim(0,100)
axs.set_xlim(0,1)


fig.savefig('loss_coloring.svg',format='svg')
# plt.show()
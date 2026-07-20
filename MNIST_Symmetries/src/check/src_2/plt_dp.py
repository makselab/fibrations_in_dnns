import pandas as pd
import matplotlib.pyplot as plt

filename = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/dL_dp_sweep_exp_01.csv'

data = pd.read_csv(filename)

fig, axs = plt.subplots(1,1)

axs.plot(data['reduction_pars_coll'],data['dL_thr'])

plt.show()
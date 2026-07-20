import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ── configuración ────────────────────────────────────────────────────────────
CSV_PATH_1  = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/dL_vs_thr_exp_03_test.csv'
CSV_PATH_2  = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/results/Evaluation_exp_03_thr_599.csv'

# ── carga ────────────────────────────────────────────────────────────────────
df_1 = pd.read_csv(CSV_PATH_1)
df_2 = pd.read_csv(CSV_PATH_2)
df_final = pd.merge(df_1, df_2, on=['thr1', 'thr2', 'thr3'])

df_final['dLsum'] = df_final['dL1']+df_final['dL2']+df_final['dL3']

# ── plot ────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(3,2, figsize=(25,25))
axs[0][0].scatter(df_final['reduction_pars_coll'], 100* df_final['dLsum']/0.10, alpha=0.3) 
axs[0][0].set_xlabel('%Compression')
axs[0][0].set_ylabel('dL/L (%)')
# axs[0][0].set_ylim([0,250])


# ── Min per %dL ─────────────────────────────────────────────────────────────
thr_per_dl = np.linspace(0,50, 201)
p = []
acc=[]
loss=[]

for thr in thr_per_dl:
   filter_1 = df_final[100*df_final['dLsum']/0.10<=thr]
   info = filter_1.loc[filter_1['reduction_pars_coll'].idxmin()]

   p.append(info['reduction_pars_coll'])
   acc.append(info['acc_coll'])
   loss.append(info['loss'])


axs[0][1].scatter(thr_per_dl, p, alpha=0.3) 
axs[0][1].set_ylabel('Optimal %Compression')
axs[0][1].set_xlabel('Threshold for dL/L (%)')
# axs[0][1].set_ylim([0,1])

axs[1][1].scatter(thr_per_dl, acc, alpha=0.3) 
axs[1][1].set_ylabel('Acc ')
axs[1][1].set_ylim([0,100])
axs[1][1].set_xlabel('Threshold for dL/L (%)')

axs[1][0].scatter(p, acc, alpha=0.3) 
axs[1][0].set_ylabel('Acc ')
axs[1][0].set_xlabel('Compression')
axs[1][0].set_ylim([0,100])
axs[1][0].set_xlim([0,1])

axs[2][0].scatter(p, np.array([100* (l-0.1)/0.1 for l in loss]), alpha=0.3) 
axs[2][0].set_xlim([0,1])
axs[2][0].set_ylim([0,900])
axs[2][0].set_ylabel('dL/L real')
axs[2][0].set_xlabel('Compression')


axs[2][1].scatter(thr_per_dl, np.array([100* (l-0.1)/0.1 for l in loss]- thr_per_dl), alpha=0.3) 
# axs[2][1].scatter(thr_per_dl, loss, alpha=0.3) 
# axs[2][1].set_xlim([0,100])
# axs[2][1].set_ylim([0,900])
axs[2][1].set_ylabel('dL/L real - dL/L approx')
axs[2][1].set_xlabel('Threshold for dL/L (%)')

fig.savefig('results_test_set.svg',format='svg')
plt.show()
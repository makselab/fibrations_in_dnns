import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name',   type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults',type=str, required=True, help='Results directory')
parser.add_argument('-epoch',      type=int, default=599,   help='Epoch')
args = parser.parse_args()

PATH = args.PATHresults + args.exp_name + '/'

comp = pd.read_csv(PATH + f'compression_loss_results_{args.epoch}.csv')
evl  = pd.read_csv(PATH + f'Evaluation_LossModels_epoch_{args.epoch}.csv')

data = comp.merge(evl, on='distance_threshold').sort_values('reduction_pars_loss')

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(data['reduction_pars_loss'], data['acc_loss'], marker='o', markersize=3)

ax.set_xlabel('Compressed Size (fraction of params)')
ax.set_ylabel('Accuracy (%)')
ax.set_xlim([0, 1])
ax.set_ylim([0, 100])
ax.grid(True, alpha=0.3)

fig.savefig(PATH + f'plots/plt_loss_accuracy_{args.epoch}.svg', format='svg')
fig.savefig(PATH + f'plots/plt_loss_accuracy_{args.epoch}.png', format='png', dpi=150)

plt.show()

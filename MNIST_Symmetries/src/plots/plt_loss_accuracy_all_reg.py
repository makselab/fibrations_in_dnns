import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name',   type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults',type=str, required=True, help='Results directory')
parser.add_argument('-epoch',      type=int, default=599,   help='Epoch')
args = parser.parse_args()

PATH = args.PATHresults + args.exp_name + '/'

fig, ax = plt.subplots(figsize=(9, 6))

# v1
comp = pd.read_csv(PATH + f'compression_loss_results_v1_{args.epoch}.csv')
evl  = pd.read_csv(PATH + f'Evaluation_LossModels_v1_epoch_{args.epoch}.csv')
data = comp.merge(evl, on='distance_threshold').sort_values('reduction_pars_loss')
ax.plot(data['reduction_pars_loss'], data['acc_loss'], marker='o', markersize=3, label='v1', color='black')

# v2 for each s_reg
reg_configs = [
    ('1e-3', 'reg_1e-3', '#1f77b4'),
    ('1e-1', 'reg_1e-1', '#ff7f0e'),
    ('0.5',  'reg_0.5',  '#2ca02c'),
    ('1',    'reg_1',    '#d62728'),
]

for label, folder, color in reg_configs:
    comp = pd.read_csv(PATH + f'{folder}/compression_loss_results_v2_{args.epoch}.csv')
    evl  = pd.read_csv(PATH + f'{folder}/Evaluation_LossModels_v2_epoch_{args.epoch}.csv')
    data = comp.merge(evl, on='distance_threshold').sort_values('reduction_pars_loss')
    ax.plot(data['reduction_pars_loss'], data['acc_loss'], marker='o', markersize=3,
            label=f'v2 s_reg={label}', color=color)

ax.set_xlabel('Compressed Size (fraction of params)')
ax.set_ylabel('Accuracy (%)')
ax.set_xlim([0, 1])
ax.set_ylim([0, 100])
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig.savefig(PATH + f'plots/plt_loss_accuracy_all_reg_{args.epoch}.svg', format='svg')
fig.savefig(PATH + f'plots/plt_loss_accuracy_all_reg_{args.epoch}.png', format='png', dpi=150)

plt.show()

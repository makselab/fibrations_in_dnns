import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
args = parser.parse_args()

configs = [
    (f'{args.exp_name}/Symmetries_thrs_[1.5, 1.5, 1.5, 0.8, 0.8, 0.3].csv', 'uniform_opfibration_0.8'),
    (f'{args.exp_name}/Symmetries_thrs_[0.8, 0.8, 0.8, 1.0, 1.0, 1.0].csv', 'uniform_fibration_0.8'),
    (f'{args.exp_name}/Symmetries_thrs_[0.75, 0.65, 0.15, 1.0, 1.0, 1.0].csv', 'optimal_fibration'),
]

acc = torch.load(args.PATHtrain + f'{args.exp_name}/accuracy.pth')
size_layer = [500, 500, 500]

for csv_file, name in configs:
    df = pd.read_csv(args.PATHresults + csv_file)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Performance')
    ax1.set_xlim([0, 600])
    ax2 = ax1.twinx()
    ax2.set_ylabel('Fibers/Num Nodes')
    ax2.set_ylim([0, 1])
    fig.tight_layout()

    ax1.plot(acc, color='green')
    for ii in range(3):
        ax2.plot(df['epoch'], df['num_colors_l'+str(ii)]/size_layer[ii], ls='--', label=str(ii))

    ax2.legend()
    fig.savefig(args.PATHresults + f'{args.exp_name}/Symmetries_vs_time_{name}.svg', format='svg')
    plt.close(fig)

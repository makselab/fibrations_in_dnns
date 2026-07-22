import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', type=str, required=True, help='Exp Name')
parser.add_argument('-PATHresults', type=str, required=True, help='Results directory')
parser.add_argument('-PATHtrain', type=str, required=True, help='Training directory')
args = parser.parse_args()

PATH_DATA = args.PATHtrain
PATH_RES  = args.PATHresults

acc = torch.load(PATH_DATA + args.exp_name + '/accuracy.pth')
trainloss = torch.load(PATH_DATA + args.exp_name + '/loss_train.pth')
testloss = torch.load(PATH_DATA + args.exp_name + '/loss_test.pth')

fig, axs= plt.subplots(2,2, figsize=(14,14))

axs[0][1].set_xscale('log')
axs[1][0].set_yscale('log')
axs[1][0].set_yscale('log')

axs[0][1].set_xlabel('Loss')
axs[0][1].set_ylabel('Accuracy')
axs[0][0].set_xlabel('Epoch')
axs[0][0].set_ylabel('Accuracy')
axs[1][0].set_ylabel('Loss')
axs[1][0].set_xlabel('Epoch')

axs[0][1].set_ylim([0,100])
axs[0][0].set_ylim([0,100])
axs[0][0].set_xlim([0,600])
axs[1][0].set_xlim([0,600])

# ---------------------------------------------------------------

axs[0][0].plot(acc, color = 'green', lw=2)
axs[1][0].plot(trainloss, color = 'blue', lw=2)
axs[1][0].plot(testloss, color = 'red', lw=2)

# ---------------------------------------------------------------
points = np.column_stack([trainloss, acc])
hull = ConvexHull(points)

hull_pts = points[hull.vertices]
hull_pts = hull_pts[np.argsort(hull_pts[:, 0])]

leftmost  = hull_pts[0]
rightmost = hull_pts[-1]

def above_baseline(p):
    t = (p[0] - leftmost[0]) / (rightmost[0] - leftmost[0] + 1e-10)
    baseline_acc = leftmost[1] + t * (rightmost[1] - leftmost[1])
    return p[1] >= baseline_acc - 1e-6

upper = np.array([p for p in hull_pts if above_baseline(p)])
upper = upper[np.argsort(upper[:, 0])]

axs[0][1].scatter(testloss, acc, alpha=0.3,color='red')
axs[0][1].scatter(trainloss, acc, alpha=0.3,color='blue')
axs[0][1].plot(testloss, acc, color='red', linewidth=2)
axs[0][1].plot(upper[1:, 0], upper[1:, 1], color='blue', linewidth=2)

# ---------------------------------------------------------------

fig.savefig(PATH_RES + args.exp_name + '/Metrics_training.svg',format='svg')

plt.show()
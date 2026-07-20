import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull

PATH_DATA = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/train_dir/'
exp_name = 'exp_03'

acc = torch.load(PATH_DATA + 'exp_03/accuracy.pth')
trainloss = torch.load(PATH_DATA + exp_name + '/loss_train.pth')
testloss = torch.load(PATH_DATA + exp_name + '/loss_test.pth')

fig, axs= plt.subplots(2,1, figsize=(7,14))
axs[0].set_yscale('log')

axs[0].scatter(acc, testloss,alpha=0.3,color='red')
axs[0].scatter(acc, trainloss,alpha=0.3,color='blue')
axs[0].plot(acc, testloss, color='red', linewidth=2)

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
axs[0].plot(upper[1:, 1], upper[1:, 0], color='blue', linewidth=2)
# ---------------------------------------------------------------

d_acc  = np.diff(upper[:, 1])
d_loss_train = np.diff(upper[:, 0])
derivative = d_loss_train / d_acc

loss_mid = (upper[:-1, 1] + upper[1:, 1]) / 2
axs[1].plot(loss_mid, derivative, color='blue', linewidth=2, marker='o', markersize=5)


d_acc  = np.diff(acc)
d_loss_test = np.diff(testloss)
derivative = d_loss_test / d_acc

loss_mid = (acc[:-1] + acc[1:]) / 2
axs[1].plot(loss_mid, derivative, color='red', linewidth=2, marker='o', markersize=5)

fig.savefig('Acc_vs_Loss.svg',format='svg')

plt.show()
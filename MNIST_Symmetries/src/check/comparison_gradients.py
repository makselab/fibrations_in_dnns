import torch
import torch.nn.functional as F

PATH = '/home/osvaldo/Documents/CCNY/MNIST_Symmetries/train_dir/exp_03/'
# Cargar gradientes
data_train = torch.load(PATH + 'gradients_training.pth')
data_test  = torch.load(PATH + 'gradients_test.pth')

grads_test = {p: torch.stack([x[p]['grad'] for x in data_test]).mean(dim=0) for p in data_test[-1].keys()}
grads_train = {p: torch.stack([x[p]['grad'] for x in data_train]).mean(dim=0) for p in data_train[-1].keys()}

for key in grads_train.keys():
    g_train = grads_train[key].flatten()
    g_test  = grads_test[key].flatten()

    cos_sim  = F.cosine_similarity(g_train.unsqueeze(0), g_test.unsqueeze(0)).item()
    rel_diff = (g_train - g_test).norm() / g_train.norm()
    ratio    = g_test.norm() / g_train.norm()

    print(f'{key}:')
    print(f'  cosine similarity : {cos_sim:.4f}')
    print(f'  relative diff     : {rel_diff:.4f}')
    print(f'  ||g_test||/||g_train|| : {ratio:.4f}')
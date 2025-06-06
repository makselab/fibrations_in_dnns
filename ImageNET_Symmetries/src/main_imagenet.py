# =================================================================
# MODULES.

import argparse
import json
import pickle
import os

import torch
from numpy import concatenate, save
from numpy.random import permutation, randint

from tqdm import tqdm
from torch.nn.functional import softmax

from models import ConvNet
from agents import agent_model
from datasets import load_imagenet
from metrics import nll_accuracy as accuracy

# =================================================================
# PARAMETERS - VARIABLES.

parser = argparse.ArgumentParser()
parser.add_argument('--cfgfilename', type=str)
parser.add_argument('--num_tasks', type=int) # 5000 
parser.add_argument('--idx_run', type=int) # 5000 
parser.add_argument('--datapath', type=str)
parser.add_argument('--respath', type=str)
args = parser.parse_args()

with open(args.cfgfilename, 'r') as f:
    params = json.load(f)

if not os.path.exists(args.respath): 
    os.makedirs(args.respath + 'weights/')
    os.makedirs(args.respath + 'performance/')

dev = torch.device("cuda:0")
idx_seq = randint(0, 299)

# ==================================================================
# DATASET

num_classes = 2
total_classes = 1000
train_images_per_class = 600
test_images_per_class = 100
train_examples_per_epoch = train_images_per_class * num_classes
test_examples_per_epoch = test_images_per_class * num_classes

with open(args.datapath + 'class_order', 'rb+') as f:
    class_order = pickle.load(f)
    class_order = class_order[idx_seq]

num_class_repetitions_required = int(num_classes * args.num_tasks / total_classes) + 1
class_order = concatenate([class_order]*num_class_repetitions_required)
save(args.respath + 'class_order.npy', class_order)

# ==================================================================
# NETWORK - AGENT

net = ConvNet(num_classes).to(dev)
agent = agent_model(net, params['optimizer'], params['agent'])

# ==================================================================
# TRAINING

num_epochs = 300
k_scalar = 2
train_mini_batch_size = 100
test_mini_batch_size = 200
save_activations_epochs = 50

train_mini_batch_size*=k_scalar
params['optimizer']['cfg']['lr']*=k_scalar 

for task_idx in range(args.num_tasks):
    # --------------------------------------------------------------

    x_train, y_train, x_test, y_test = load_imagenet(args.datapath,
                                                    train_images_per_class, test_images_per_class,
                                                    class_order[task_idx*num_classes:(task_idx+1)*num_classes]
                                                    )

    x_train, x_test, y_train, y_test = x_train.to(dev), x_test.to(dev), y_train.to(dev), y_test.to(dev)

    net.layers[-1].weight.data *= 0
    net.layers[-1].bias.data   *= 0

    train_accuracy = torch.zeros(num_epochs)
    test_accuracy  = torch.zeros(num_epochs)

    # --------------------------------------------------------------

    for epoch_idx in tqdm(range(num_epochs)):

        # Evaluation -----------------------------------------------

        test_accuracies = []
        #list_x = []
        #list_y = []
        #list_activations = []
        #list_outputs = []

        net.eval()

        for start_idx in range(0, test_examples_per_epoch, test_mini_batch_size):
            test_batch_x = x_test[start_idx: start_idx + test_mini_batch_size]
            test_batch_y = y_test[start_idx: start_idx + test_mini_batch_size]
            #list_x.append(test_batch_x)
            #list_y.append(test_batch_y)

            outputs, activations = net(x=test_batch_x)
            #list_outputs.append(outputs)
            #list_activations.append(activations)

            test_accuracies.append(accuracy(softmax(outputs, dim=1), test_batch_y))

            loss = agent.loss_func(outputs, test_batch_y)
            loss.backward()

        test_accuracy[epoch_idx] = sum(test_accuracies)/len(test_accuracies)    

        #deltas = net.get_deltas()

        # Save activations -----------------------------------------
        if epoch_idx % save_activations_epochs == 0:
            # with open(args.respath + 'activations_errors/task_idx_' + str(task_idx) + '_epoch_' + str(epoch_idx) + '.pkl' , 'wb+') as f:
            #     pickle.dump(obj={
            #         'x': list_x,
            #         'y': list_y,
            #         'activity': list_activations,
            #         'outputs': list_outputs,
            #         'errors': deltas,
            #         }, file = f)

            torch.save(net.state_dict(), 
                        args.respath + 'weights/task_idx_' + str(task_idx) + '_epoch_' + str(epoch_idx) + '.pth')

        # Training -------------------------------------------------
        #net.eval()
        net.train()

        example_order = permutation(train_images_per_class * num_classes)
        x_train = x_train[example_order]
        y_train = y_train[example_order]

        train_accuracies = []

        for start_idx in range(0, train_examples_per_epoch, train_mini_batch_size):
            batch_x = x_train[start_idx: start_idx+train_mini_batch_size]
            batch_y = y_train[start_idx: start_idx+train_mini_batch_size]

            loss, outputs = agent.learn(x=batch_x, target=batch_y)
            train_accuracies.append(accuracy(softmax(outputs, dim=1), batch_y))

        train_accuracy[epoch_idx] = sum(train_accuracies)/len(train_accuracies)

    # Save performances -----------------------------------------------
    with open(args.respath + 'performance/task_idx_' + str(task_idx) + '.pkl' , 'wb+') as f:
        pickle.dump(obj={
                    'Train': train_accuracy,
                    'Test': test_accuracy}, file = f)

    del x_train, x_test, y_train, y_test
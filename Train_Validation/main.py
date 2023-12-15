from lib.networkNS import NetComNS_InN_legendre
from lib.utils import Initialization_factors
from lib.train import train_iterative_ComNS_InN as train
from lib.test import test_iterative_ComNS_InN as test
from Data.DatasetNS import ComNS_Dataset2D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import math
import scipy.io as scio
import numpy as np
import cupy as cp
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt

PROBLEM = 'ComNS'

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--out_name", help="name of this try")
args = parser.parse_args()

def get_parameter_number(network):
    total_num = sum(p.numel() for p in network.parameters())
    trainable_num = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

in_length = 1

# training parameters
learning_rate = 0.001
weight_decay = 1e-4
lr_min = 0
momentum = 0.9
batch_size = 2
print_frequency = 50

rounds = 10
epochs = 20  # 15
epochs_overall = rounds * epochs

# learning task
Re = 100
Ma = 2  # *0.1
t_interval = 5

# network parameters
N = 12
K = 2
M = 6
num_blocks = 4
Params = {
    'n': N,
    'm': M,
    'k': K,
    'norm_factors': [0.5, 0.5, 5, 10],  # [1, 1, 1, 1], #
    'init_weight': [math.sqrt(3), math.sqrt(Initialization_factors['({},{},{})'.format(N, K, M)]),
                    math.sqrt(6), math.sqrt(3), math.sqrt(6), math.sqrt(3)],  # [1,1,1,1,1,1], #
    'if_ln': True  # False
}

def train_test_save():
    orders_all = [i for i in range(1, 31)]  # sample dataset
    orders_test = [i for i in range(1, 26)]
    '''if Re == 20:
        orders_all = [i for i in range(1, 451)] #Re20Ma2
        orders_test = [i for i in range(1, 51)]
    else:
        orders_all = [i for i in range(1, 31)]
        orders_test = [i for i in range(1, 26)]'''

    orders_train = [order for order in orders_all if order not in orders_test]

    print('orders_train = ', orders_train)
    print('orders_test = ', orders_test)

    dataset = ComNS_Dataset2D(data_dir='Data/', data_name='ComNS128Re{}Ma{}'.format(Re, Ma), orders_train=orders_train,
                              orders_test=orders_test,
                              t_interval=t_interval,
                              chara_velocity = 'c',
                              if_ln=Params['if_ln'])

    network = NetComNS_InN_legendre(num_blocks=num_blocks,Params=Params)

    print(dataset.cache_names)
    print(dataset.sample_nums)
    print(network)
    print(get_parameter_number(network))

    network = network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=momentum,
    #                             weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7, last_epoch=-1)

    recurrent = 10
    for r in range(rounds):
        train_generator = dataset.data_generator_series(out_length=recurrent, in_length=in_length, batch_size=batch_size)
        train(network=network, batch_size=batch_size,
              epochs=epochs, max_ep=epochs_overall, last_ep=r * epochs, optimizer=optimizer,
              dataset=dataset, train_gen=train_generator, round=recurrent,
              print_frequency=print_frequency,
              In_length=in_length)
        scheduler.step()

    torch.save(network, 'models/' + args.out_name + '_model.pp')

def load_test(out_name):
    model_file = out_name + '_model.pp'
    network = torch.load('models/' + model_file)
    print(network)
    network = network.cuda()

    if Re == 20:
        orders_test = [i for i in range(1, 51)]
    else:
        orders_test = [i for i in range(1, 26)]

    ground_truth_name = 'ComNS128Re{}Ma{}'.format(Re,Ma)

    dataset = ComNS_Dataset2D(data_dir='Data/', data_name='ComNS128Re{}Ma{}'.format(Re,Ma), orders_train=[],
                             orders_test=orders_test,
                             t_interval=t_interval,
                              chara_velocity = 'c',
                              if_ln=Params['if_ln'])

    test_round = 10
    long_round = int(500 / t_interval)
    test_generator = dataset.data_generator_series(out_length=test_round, in_length=in_length, batch_size=batch_size,
                                                   split='test')
    test(network, dataset, test_generator, test_round, long_round, args.out_name, In_length=in_length,
             if_ln=Params['if_ln'])

    MeanSquareError(out_name, ground_truth_name, orders_test)

def MeanSquareError(out_name, ground_truth_name, orders_test):
    output = scio.loadmat('outputs/' + out_name + '.mat')['output']
    output[:, :, 2:4, :, :] = np.exp(output[:, :, 2:4, :, :])
    NG = output.shape[-1]
    if Re == 20:
        test_t = [int(20 / t_interval) - 1, int(50 / t_interval) - 1, int(100 / t_interval) - 1,
                  int(200 / t_interval) - 1]  # , int(500/t_interval)-1]
        t_max = int(250 / t_interval)
    else:
        test_t = [int(20 / t_interval) - 1, int(50 / t_interval) - 1, int(100 / t_interval) - 1,
                  int(200 / t_interval) - 1, int(500/t_interval)-1]
        t_max = int(500 / t_interval)
    MSE = np.zeros(len(test_t))
    MSE_all = [np.zeros(t_max), np.zeros(t_max), np.zeros(t_max)] # velocity, rho, T
    for k in range(len(orders_test)):
        order = orders_test[k]
        ground_truth = scio.loadmat('Data/' + ground_truth_name + '/' + ground_truth_name + '_' + str(order) + '.mat')
        i = 0
        for ii in range(t_max):
            ground_truth_u = ground_truth['u'][(ii+1) * t_interval, :].reshape(NG, NG)
            ground_truth_v = ground_truth['v'][(ii+1) * t_interval, :].reshape(NG, NG)
            ground_truth_rho = ground_truth['rho'][(ii + 1) * t_interval, :].reshape(NG, NG)
            ground_truth_T = ground_truth['T'][(ii + 1) * t_interval, :].reshape(NG, NG)
            MSE_all[0][ii] += np.mean(np.sqrt((output[ii, k, 0, :, :] - ground_truth_u) ** 2 + (
                        output[ii, k, 1, :, :] - ground_truth_v) ** 2))
            MSE_all[1][ii] += np.mean(np.sqrt((output[ii, k, 2, :, :] - ground_truth_rho) ** 2))
            MSE_all[2][ii] += np.mean(np.sqrt((output[ii, k, 3, :, :] - ground_truth_T) ** 2))
            if ii in test_t:
                MSE[i] += np.mean(np.sqrt((output[ii, k, 0, :, :] - ground_truth_u) ** 2 + (
                        output[ii, k, 1, :, :] - ground_truth_v) ** 2))
                i += 1
    for i in range(len(MSE)):
        MSE[i] = MSE[i]/len(orders_test)
    print('MSE(t=0.2, 0.5, 1, 2, 5)={}'.format(MSE))

    for i in range(len(MSE_all)):
        MSE_all[i] = MSE_all[i] / len(orders_test)
    print('averaged MSE [uv,rho,T] = {}'.format([np.mean(MSE_all[0]),np.mean(MSE_all[1]),np.mean(MSE_all[2])]))

    with open('MSE_t/{}_MSE.log'.format(args.out_name), 'w') as f:
        f.write('Error of UV:\n')
        for mse in MSE_all[0]:
            f.write('{}\n'.format(mse))
        f.write('\n')
        f.write('Error of rho:\n')
        for mse in MSE_all[1]:
            f.write('{}\n'.format(mse))
        f.write('\n')
        f.write('Error of T:\n')
        for mse in MSE_all[2]:
            f.write('{}\n'.format(mse))

    print('test MSE of {} checkpoints saved at MSE_t/{}_MSE.log'.format(t_max, args.out_name))

def test_receptive(network_name):
    #network = NetComNS_InN_legendre(num_blocks=4,Params=Params)
    model_file = 'models/{}_model.pp'.format(network_name)
    network = torch.load(model_file)

    network.cuda()

    k_x = 64
    k_y = 64

    channel = 4
    norm_grad = torch.zeros(128, 128)
    grad_decomposed = torch.zeros(channel, channel, 128, 128)
    Nk = int(network.n / network.k)

    round = 100
    for a in range(Nk):
        for b in range(Nk):
            grad = torch.zeros(round, channel, channel, 128, 128).cuda()
            for i in range(round):
                input = Variable(torch.ones(1, channel, 128, 128), requires_grad=True) * torch.randn(1) * 0.05
                for c in range(channel):
                    input[:, c, :, :] = input[:, c, :, :] / network.norm_factors[c]
                input = input.cuda()
                input.retain_grad()

                for c_out in range(channel):
                    output = network(input)

                    output_sum = torch.sum(output[:, c_out, k_x // Nk * Nk + a, k_y // Nk * Nk + b])

                    network.zero_grad()
                    input.grad = torch.zeros(1, channel, 128, 128).cuda()
                    output_sum.backward(retain_graph=True)

                    grad_temp = input.grad[0, :, :, :]
                    grad[i,c_out,:,:,:] = grad[i,c_out,:,:,:] + grad_temp #grad_temp[0,c_in,:,:] #torch.abs(grad_temp[0,c_in,:,:])

            grad = grad.cpu()

            # for channel decomposition
            grad_decomposed[:, :, :-a - 1, :-b - 1] = grad_decomposed[:, :, :-a - 1, :-b - 1] + grad[0, :, :, a:-1, b:-1]

            # mixed receptive field
            grad = torch.std(grad,dim=0)
            norm_grad[:-a-1,:-b-1] = norm_grad[:-a-1,:-b-1] + torch.mean(grad, (0,1))[a:-1,b:-1]
            print('{},{}'.format(a, b))

    '''
    dict = {}
    for c_in in range(channel):
        for c_out in range(channel):
            rcp_field = grad_decomposed[c_in, c_out, :, :]
            plt.imsave('receptives/{}_{}-{}_receptive.png'.format(network_name, c_in, c_out), rcp_field, cmap=plt.cm.gray)
            print('receptive figure saved!')
            dict['grad_D{}D{}'.format(c_out,c_in)] = rcp_field.numpy()
    scio.savemat('receptives/{}_receptive_decomposed'.format(network_name), mdict=dict)
    print('decomposed receptive mat saved!')
    '''

    scio.savemat('receptives/{}_receptive'.format(network_name), mdict={"grad": norm_grad.numpy()})
    print('receptive mat saved!')

    rcp_field = norm_grad
    plt.imsave('receptives/{}_receptive.png'.format(network_name), rcp_field, cmap=plt.cm.gray)
    print('receptive figure saved!')

if __name__ == '__main__':
    train_test_save()
    #load_test(args.out_name)
    #test_receptive(args.out_name)

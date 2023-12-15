import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio

Initialization_factors={
'(6,2,4)': 10.3146,
'(6,2,6)': 7.5625,
'(8,2,6)': 7.7491,
'(10,2,6)': 10.1948,
'(12,2,6)': 11.8781,
'(12,3,6)': 35.4591,
'(12,4,6)': 68.4968,
'(12,6,6)': 176.1236,
'(12,12,6)': 358.5074,
'(12,2,2)': 252.3647,
'(12,2,4)': 41.3953,
'(12,2,8)': 5.1523,
'(12,2,10)': 3.2647,
'(12,2,12)': 2.6566,
'(14,2,6)': 18.5122,
'(16,2,6)': 25.5560,
'(18,2,6)': 31.9326,
'(20,2,6)': 37.6120,
'(24,2,6)': 37.6120, #53.2937,
'(28,2,6)': 37.6120, #73.1328,
}

def generte_legendre_filters_2D(file, n, m):
    '''
    :param file: the path of the legendre filters
    :param n: the length of the legendre filters
    :param m: the number of modes reserved
    :return: the filters for decomposition and reconstruction, the length of the filters
    '''
    filters_1D = scio.loadmat(file.format(n))

    window = np.append(np.linspace(0, 1, n // 2), np.linspace(1, 0, n // 2))


    filters_forward = []
    filters_backward = []
    for i in range(m):
        filters_forward.append(filters_1D['forward'][i,:])

        filters_backward.append(filters_1D['backward'][i,:])
        assert len(filters_1D['forward'][i,:]) == n

    filters_d = []
    filters_r = []
    for i in range(m):
        filters_forward[i] =  torch.Tensor(filters_forward[i]).reshape(1, n)# * window
        filters_backward[i] = torch.Tensor(filters_backward[i]).reshape(1, n)

    for i in range(m):
        for j in range(m):
            filters_d += [torch.mul(filters_forward[i].T, filters_forward[j]).reshape(1, 1, n, n)]
            filters_r += [torch.mul(filters_backward[i].T, filters_backward[j]).reshape(1, 1, n, n)]

    filter_d = torch.cat(tuple(filters_d), dim=0)
    filter_r = torch.cat(tuple(filters_r), dim=0)
    return filter_d, filter_r, n

def generte_legendre_filters_1D(file, n, m):
    '''
    :param file: the path of the legendre filters
    :param n: the length of the legendre filters
    :param m: the number of modes reserved
    :return: the filters for decomposition and reconstruction, the length of the filters
    '''
    filters_1D = scio.loadmat(file.format(n))

    window = np.append(np.linspace(0, 1, n // 2), np.linspace(1, 0, n // 2))


    filters_forward = []
    filters_backward = []
    for i in range(m):
        filters_forward.append(filters_1D['forward'][i,:])
        filters_backward.append(filters_1D['backward'][i,:])
        assert len(filters_1D['forward'][i,:]) == n

    filters_d = []
    filters_r = []
    for i in range(m):
        filters_forward[i] =  torch.Tensor(filters_forward[i]).reshape(1, n)# * window
        filters_backward[i] = torch.Tensor(filters_backward[i]).reshape(1, n)

    for i in range(m):
        filters_d += [filters_forward[i].reshape(1, 1, n,)]
        filters_r += [filters_backward[i].reshape(1, 1, n)]

    filter_d = torch.cat(tuple(filters_d), dim=0)
    filter_r = torch.cat(tuple(filters_r), dim=0)
    return filter_d, filter_r, n

def generte_boundary_filters_2D(file, n, m, normal_N, normal_m):
    '''
    :param file: the path of the legendre filters
    :param n: the length of the legendre filters
    :param m: the number of modes reserved
    :param normal_N: the length of the legendre filters (normal direction)
    :param normal_m: the number of modes reserved (normal direction)
    :return: the filters for decomposition and reconstruction, the length of the filters
    '''
    filters_1D_tangential = scio.loadmat(file.format(n))
    filters_1D_normal = scio.loadmat(file.format(normal_N))

    #window = np.append(np.linspace(0, 1, n // 2), np.linspace(1, 0, n // 2))

    filters_forward_tangential = []
    filters_backward_tangential = []
    filters_forward_normal = []
    filters_backward_normal = []
    for i in range(m):
        filters_forward_tangential.append(filters_1D_tangential['forward'][i,:])
        filters_backward_tangential.append(filters_1D_tangential['backward'][i,:])
        assert len(filters_1D_tangential['forward'][i, :]) == n
        filters_forward_tangential[i] = torch.Tensor(filters_forward_tangential[i]).reshape(n, 1)
        filters_backward_tangential[i] = torch.Tensor(filters_backward_tangential[i]).reshape(n, 1)

    for i in range(normal_m):
        filters_forward_normal.append(filters_1D_normal['forward'][i,:])
        filters_backward_normal.append(filters_1D_normal['backward'][i,:])
        assert len(filters_1D_normal['forward'][i,:]) == normal_N
        filters_forward_normal[i] = torch.Tensor(filters_forward_normal[i]).reshape(1, normal_N)
        filters_backward_normal[i] = torch.Tensor(filters_backward_normal[i]).reshape(1, normal_N)

    filters_d = []
    filters_r = []

    for i in range(m):
        for j in range(normal_m):
            filters_d += [torch.mul(filters_forward_tangential[i], filters_forward_normal[j]).reshape(1, 1, n, normal_N)]
            filters_r += [torch.mul(filters_backward_tangential[i], filters_backward_normal[j]).reshape(1, 1, n, normal_N)]

    filter_d = torch.cat(tuple(filters_d), dim=0)
    filter_r = torch.cat(tuple(filters_r), dim=0)
    return filter_d, filter_r, n

def spatial_gradient(Tensor, channels = None):
    Tensor_xshift = torch.zeros(Tensor.shape).cuda()
    Tensor_yshift = torch.zeros(Tensor.shape).cuda()

    if channels is None:
        channel = Tensor.shape[1]
        channels = [i for i in range(channel)]

    for i in channels:
        Tensor_xshift[:, i, 1:, :] = Tensor[:, i, :-1, :]
        Tensor_xshift[:, i, :1, :] = Tensor[:, i, -1:, :]
        Tensor_yshift[:, i, :, 1:] = Tensor[:, i, :, :-1]
        Tensor_yshift[:, i, :, :1] = Tensor[:, i, :, -1:]

    dx = 1/64

    Tensor_sgX = (Tensor_xshift - Tensor)/dx
    Tensor_sgY = (Tensor_yshift - Tensor)/dx

    return torch.cat([Tensor_sgX, Tensor_sgY], dim=1)
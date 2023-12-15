import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import gc
import argparse
import math
import time

from IBM import *

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--out_name", help="name of this try")
args = parser.parse_args()

with torch.no_grad():
    N = 12
    if N == 8:
        model_file = 'ComNS_Re100Ma2_n4N8m6k2_interval50_model.pp'
    elif N == 12:
        model_file = 'ComNS_Re100Ma2_n4N12m6k2_interval50_model.pp'
    elif N == 20:
        model_file = 'ComNS_Re100Ma2_n4N20m6k2_interval50_model.pp'

    network = torch.load('models/' + model_file)
    network.eval()

    L = 64
    NG = L
    NG_L = 10*L
    NG_D = L*10
    NG_U = L*10
    NG_R = 30*L
    Length_x = NG_L+NG_R+NG
    Length_y = NG_U+NG_D
    delta_x = 2/128

    u_lid = 1

    Length_x += network.get_padding_R(Length_x+2*network.recept + 1)
    AUX_y = network.get_padding_R(Length_y+2*network.recept)

    NewBoundaryTreat = True

    xMin = 0
    xMax = xMin + delta_x * NG
    yMin = 0
    yMax = yMin + delta_x * NG
    print('length_x=',Length_x)
    print('length_y=',Length_y)

    delta_t = 0.05
    Re = 100
    Ma = 0.2
    R_fluid = 1/1.4/Ma/Ma
    t = 0

    '''计算IBM节点坐标'''
    filename = 'CircularCylinderD10.mat'
    '''数据文件格式：
    p_l:n*2数组 顺时针记录边界Lagrange点坐标，第一个点出现2次以实现封闭，从坐标原点开始
    delta_s:Lagrange点之间的距离'''
    this_raw_data = scio.loadmat('geometry/'+filename)
    delta_s = this_raw_data['delta_s']
    ShapeNum = int(this_raw_data['ShapeNum'])
    d = this_raw_data['d'][0,0]

    p_l = []
    for i in range(ShapeNum):
        p_l.append(this_raw_data['p_l' + str(i)])
        p_l[i][:, 0] += xMin + NG_L * delta_x - d / 2
        p_l[i][:, 1] += yMin + Length_y / 2 * delta_x

    p_e = np.zeros((Length_y, Length_x + 1, 2), dtype='float32')
    for i in range(Length_y):
        for j in range(Length_x + 1):
            p_e[i, j, 0] = j * delta_x + xMin
            p_e[i, j, 1] = i * delta_x + yMin
    p_e = p_e.reshape([(Length_y) * (Length_x + 1), 2])

    IBM = ClassicIBM(delta_x,p_e,p_l,p_l,Length_x,Length_y-1,delta_s=delta_s)

    u_NN = np.ones((Length_y, Length_x + 1), dtype='float32')*u_lid
    v_NN = np.zeros((Length_y, Length_x + 1), dtype='float32')
    rho_NN = np.zeros((Length_y, Length_x + 1), dtype='float32')
    T_NN = np.zeros((Length_y, Length_x + 1), dtype='float32')

    u_NN = torch.from_numpy(u_NN).cuda()
    v_NN = torch.from_numpy(v_NN).cuda()
    rho_NN = torch.from_numpy(rho_NN).cuda()
    T_NN = torch.from_numpy(T_NN).cuda()

    r = network.recept + 0
    output = torch.zeros(1, 4, Length_y, Length_x + 1).cuda()
    cycle_num = 201


    # zero shifting correction
    u_NN = F.pad(u_NN, (r, r, r + AUX_y, r), mode='constant', value=u_lid)
    v_NN = F.pad(v_NN, (r, r, r + AUX_y, r), mode='constant', value=0)
    rho_NN = F.pad(rho_NN, (r, r, r + AUX_y, r), mode='constant', value=0)
    T_NN = F.pad(T_NN, (r, r, r + AUX_y, r), mode='constant', value=0)
    input = torch.stack((u_NN, v_NN, rho_NN, T_NN))
    input = torch.unsqueeze(input, 0)
    output = network(input)[:, :, AUX_y:, :]
    u_NN = output[0, 0, :, :]
    v_NN = output[0, 1, :, :]
    rho_NN = output[0, 2, :, :]
    T_NN = output[0, 3, :, :]
    u_correction = (torch.mean(u_NN)-u_lid )*0.9
    v_correction = (torch.mean(v_NN))*0.9
    rho_correction = (torch.mean(rho_NN))*0.9
    T_correction = (torch.mean(T_NN))*0.9

    t1 = time.time()

    for cycle in range(cycle_num):

        input0 = torch.stack((u_NN, v_NN, rho_NN, T_NN))

        u_NN = F.pad(u_NN, (r, r, r + AUX_y, r), mode='constant', value=u_lid)
        v_NN = F.pad(v_NN, (r, r, r + AUX_y, r), mode='constant', value=0)
        rho_NN = F.pad(rho_NN, (r, r, r + AUX_y, r), mode='constant', value=0)
        T_NN = F.pad(T_NN, (r, r, r + AUX_y, r), mode='constant', value=0)
        input = torch.stack((u_NN, v_NN, rho_NN, T_NN))
        input = torch.unsqueeze(input, 0)

        output = network(input)[:, :, AUX_y:, :]

        u_NN = output[0, 0, :, :]
        v_NN = output[0, 1, :, :]
        rho_NN = output[0, 2, :, :]
        T_NN = output[0, 3, :, :]


        if NewBoundaryTreat:
            u_NN, v_NN, rho_NN, T_NN = IBM.Newstep(output)
        else:
            u_NN, v_NN, rho_NN, T_NN = IBM.step(output)

        u_NN = u_NN - u_correction
        v_NN = v_NN - v_correction
        rho_NN = rho_NN - rho_correction
        T_NN = T_NN - T_correction
        if cycle%20==0:
            print(cycle,'/',cycle_num)

            scio.savemat('CircularCylinder/CC__' + str(cycle) + '.mat', mdict={'u': u_NN.cpu().detach().numpy(),
                                                                  'v': v_NN.cpu().detach().numpy(),
                                                                  'rho': rho_NN.cpu().detach().numpy(),
                                                                  'T': T_NN.cpu().detach().numpy()})

        gc.collect()





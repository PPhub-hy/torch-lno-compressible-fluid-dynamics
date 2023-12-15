import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import gc
import argparse
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

    L =32
    NG = L
    NG_L = 16*L
    NG_R = 100*L
    Length_x = NG_L+NG_R
    Length_y = 13*L
    delta_x = 2/128
    NewBoundaryTreat = True

    u_lid = 1

    AUX_x = network.get_padding_R(Length_x+2*network.recept+1)
    AUX_y = network.get_padding_R(Length_y+2*network.recept+1)

    xMin = -NG_L *delta_x
    xMax = xMin + delta_x * Length_x
    yMin = 0
    yMax = yMin + delta_x * Length_y
    print('length_x=',Length_x)
    print('length_y=',Length_y)
    print('AUX_x=', AUX_x)
    print('AUX_y=', AUX_y)
    t_interval = 5
    delta_t = 0.01 * t_interval
    Re = 100
    Ma = 0.2
    R_fluid = 1/1.4/Ma/Ma
    t = 0
    alpha = -0/180*np.pi

    '''计算IBM节点坐标'''
    filename = 'vehicle_sports.mat'
    '''数据文件格式：
    p_l:n*2数组 顺时针记录边界Lagrange点坐标，第一个点出现2次以实现封闭，从坐标原点开始'''
    this_raw_data = scio.loadmat('geometry/'+filename)
    p_e = np.zeros((Length_y+1, Length_x + 1,2), dtype='float32')
    for i in range(Length_y+1):
        for j in range(Length_x+1):
            p_e[i,j,0]=j*delta_x+xMin
            p_e[i, j, 1] = i * delta_x + yMin
    p_e = p_e.reshape([(Length_y+1)*(Length_x + 1),2])
    ShapeNum = int(this_raw_data['ShapeNum'])
    p_l = []
    for i in range(ShapeNum):
        p_l.append(this_raw_data['p_l' + str(i)])
        p_l[i][:, 0] += xMin + NG_L * delta_x
        p_l[i][:, 1] += yMin
    p_ori = []
    for i in range(ShapeNum):
        p_ori.append(this_raw_data['p_ori' + str(i)])
        p_ori[i][:, 0] += xMin + NG_L * delta_x
        p_ori[i][:, 1] += yMin
    if not NewBoundaryTreat:
        delta_s = this_raw_data['delta_s']
        IBM = ClassicIBM(delta_x, p_e, p_l, p_ori, Length_x, Length_y, delta_s=delta_s, NewBoundaryTreatment=NewBoundaryTreat, If_Com=True)
    else:
        IBM = ClassicIBM(delta_x, p_e, p_l, p_ori, Length_x, Length_y, NewBoundaryTreatment=NewBoundaryTreat, If_Com=True)

    u_NN = np.ones((Length_y+1, Length_x + 1), dtype='float32')*u_lid*np.cos(alpha)
    v_NN = np.ones((Length_y+1, Length_x + 1), dtype='float32')*u_lid*np.sin(alpha)
    rho_NN = np.zeros((Length_y+1, Length_x + 1), dtype='float32')
    T_NN = np.zeros((Length_y+1, Length_x + 1), dtype='float32')

    u_NN = torch.from_numpy(u_NN).cuda()
    v_NN = torch.from_numpy(v_NN).cuda()
    rho_NN = torch.from_numpy(rho_NN).cuda()
    T_NN = torch.from_numpy(T_NN).cuda()

    u_NN[0, :] = 0

    r = network.recept
    output = torch.zeros(1, 4, Length_y+1, Length_x + 1).cuda()

    cycle_num = 101
    F_d_his = np.zeros(cycle_num, dtype='float32')
    F_l_his = np.zeros(cycle_num, dtype='float32')

    # 零漂修正
    u_correction = (torch.mean(u_NN) - u_lid * np.cos(alpha)) * 0.9
    v_correction = (torch.mean(v_NN) - u_lid * np.sin(alpha)) * 0.9
    rho_correction = (torch.mean(rho_NN)) * 0.9
    T_correction = (torch.mean(T_NN)) * 0.9

    t1 = time.time()

    for cycle in range(cycle_num):

        input1 = torch.stack((u_NN, v_NN))
        input1 = torch.unsqueeze(input1, 0)
        input1 = F.pad(input1, (r+AUX_x, r, r+AUX_y, r), mode='replicate')
        input2 = torch.stack((rho_NN, T_NN))
        input2 = torch.unsqueeze(input2, 0)
        input2 = F.pad(input2, (r+AUX_x, r, 0, 0), mode='replicate')
        input2 = F.pad(input2, (0, 0, r+AUX_y, r), mode='reflect')
        input = torch.cat((input1, input2),dim=1)

        output = network(input[:, :, :, :])[:, :, AUX_y:, AUX_x:]

        if NewBoundaryTreat:
            u_NN, v_NN, rho_NN, T_NN = IBM.Newstep(output)
        else:
            u_NN, v_NN, rho_NN, T_NN = IBM.step(output)


        # u_NN = u_NN - u_correction
        # v_NN = v_NN - v_correction
        # rho_NN = rho_NN - rho_correction
        # T_NN = T_NN - T_correction

        u_NN[:, 0] = u_lid
        u_NN[:, -1] = u_lid
        u_NN[0, :] = u_lid
        u_NN[-1, :] = u_lid


        v_NN[:, 0] = 0
        v_NN[:, -1] = 0
        v_NN[0, :] = 0
        v_NN[-1, :] = 0

        T_NN[:, 0] = 0
        T_NN[:, -1] = 0

        rho_NN[:, 0] = 0
        rho_NN[:, -1] = 0


        if cycle%20==0:
            print(cycle,'/',cycle_num)
            scio.savemat('Vehicle/Vehicle_' + str(cycle) + '.mat', mdict={'u': u_NN.cpu().detach().numpy(),
                                                                              'v': v_NN.cpu().detach().numpy(),
                                                                              'rho': rho_NN.cpu().detach().numpy(),
                                                                              'T': T_NN.cpu().detach().numpy()})

        gc.collect()

    torch.cuda.synchronize()
    t2 = time.time()
    print('t2-t1=', t2 - t1)




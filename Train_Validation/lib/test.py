import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

def test_iterative_ComNS_InN(network, dataset, test_gen, round, long_round, out_name, In_length, if_ln=False):
    N = 128
    length_in = In_length

    network.eval()
    data_input, data_output = next(test_gen)

    with torch.no_grad():

        record_round = 1
        NG = data_input.shape[3]
        inp = dataset.load_test_input(length_in)
        inp = inp.cuda()
        if if_ln:
            inp[:, 2:4, :, :] = torch.log(inp[:, 2:4, :, :])

        data_matlab = torch.zeros(long_round // record_round, inp.shape[0], 4, NG, NG)
        for i in range(long_round):
            output = network(inp)
            if i % record_round == 0:
                data_matlab[i // record_round, :, :, :, :] = output[:, :, :, :]
            new_in = torch.zeros(inp.shape[0], 4 * length_in, NG, NG).cuda()
            home = 4 * (length_in - 1)
            new_in[:, :home, :, :] = inp[:, 4:, :, :]
            new_in[:, home:home + 4, :, :] = output
            inp = new_in

    scio.savemat('outputs/'+out_name, mdict={"output": data_matlab.detach().numpy()})

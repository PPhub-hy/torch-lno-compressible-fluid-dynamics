import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import scipy.io as scio
import math
from lib.utils import generte_legendre_filters_2D, generte_legendre_filters_1D, generte_boundary_filters_2D
from lib.utils import spatial_gradient, Initialization_factors

class NetComNS_InN_legendre(nn.Module):

    def __init__(self, num_blocks, Params=None, cheb=False):
        super(NetComNS_InN_legendre, self).__init__()
        self.num_blocks = num_blocks
        with_bias = False  # True#

        self.filter_d = None
        self.filter_r = None

        self.norm_factors = Params['norm_factors']
        self.if_ln = Params['if_ln']
        print('norm_factors:{} (u,v,rho,T)'.format(self.norm_factors))
        print('if_ln:{}', format(self.if_ln))
        self.n = Params['n']
        self.m = Params['m']
        self.k = Params['k']
        print('legendre params: n:{} m:{} k:{}'.format(self.n, self.m, self.k))
        if cheb:
            file_path = 'lib/chebyshevs/ChebyshevConv{}.mat'
            print("*" * 5, 'using Chebyshev filters', "*" * 5)
        else:
            file_path = 'lib/legendres/LegendreConv{}.mat'
            print("*" * 5, 'using Legendre filters', "*" * 5)
        self.filter_d, self.filter_r, self.l_filters = generte_legendre_filters_2D(file_path,
                                                                                   n=self.n,
                                                                                   m=self.m)
        self.modes = self.filter_d.shape[0]

        self.convs = []
        self.WPDlayers = []
        self.linears = []
        self.linearModes = []

        first_channel = 1
        out_channel = 1

        channel = 40
        self.first_channel = first_channel
        self.channel = channel
        self.first_conv_kernel_size = 0
        self.first_conv = nn.Conv2d(first_channel, channel, kernel_size=2 * self.first_conv_kernel_size + 1,
                                    bias=with_bias)
        for i in range(num_blocks):
            self.linearModes.append(
                nn.Conv2d(self.modes * channel, self.modes * channel, kernel_size=1, groups=channel, bias=with_bias)
            )
            print('spectral layer {}, modes {}'.format(i + 1, self.modes))

            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, dilation = 1, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, dilation = 1, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                    nn.GELU(),
                    nn.Conv2d(channel, channel, kernel_size=3, bias=with_bias),
                )
            )
        lifting = 128
        self.conv11 = nn.Conv2d(channel, lifting, kernel_size=1, bias=with_bias)
        self.convout = nn.Conv2d(lifting, out_channel, kernel_size=1, bias=with_bias)

        # Normalize the initial weights
        self.init_weight = Params['init_weight']

        self.first_conv.weight = nn.Parameter(self.first_conv.weight * self.init_weight[0])
        for i in range(num_blocks):
            self.linearModes[i].weight = nn.Parameter(self.linearModes[i].weight * self.init_weight[1])
            self.convs[i][0].weight = nn.Parameter(self.convs[i][0].weight * self.init_weight[2])
            self.convs[i][2].weight = nn.Parameter(self.convs[i][2].weight * self.init_weight[2])
            self.convs[i][4].weight = nn.Parameter(self.convs[i][4].weight * self.init_weight[2])
            self.convs[i][-1].weight = nn.Parameter(self.convs[i][-1].weight * self.init_weight[3])

        self.conv11.weight = nn.Parameter(self.conv11.weight * self.init_weight[4])
        self.convout.weight = nn.Parameter(self.convout.weight * self.init_weight[5])

        print('init_weight=:sqrt{} (first_conv, LinearModes, convs1, convs2, conv11, convout)'.format(
            np.array(self.init_weight) ** 2))

        self.convs = nn.ModuleList(self.convs)
        self.linears = nn.ModuleList(self.linears)
        self.linearModes = nn.ModuleList(self.linearModes)

        self.RR = self.n // self.k * (self.k - 1)
        self.recept = self.RR * self.num_blocks + self.first_conv_kernel_size
        print('Range of receptive domain: {}'.format(self.recept))

        self.filter_r = self.filter_r.cuda()
        # self.filter_r = torch.flip(self.filter_r, (2,3))
        self.filter_d = self.filter_d.cuda()

    def get_padding_R(self, l_input):
        n = self.n
        if l_input < n:
            return n - l_input

        assert n % self.k == 0
        n = n // self.k * (self.k - 1)

        return (n - l_input % n) % n

    def forward(self, input):
        input_ln = input.clone()

        interior_tensor_output = []

        recept = self.recept
        r = self.get_padding_R(input_ln.shape[-1] + 2 * recept - 2 * self.first_conv_kernel_size)
        left = recept
        right = recept + r
        input_ln = F.pad(input_ln, (left, right, left, right), "circular")

        interior_tensor_output.append(input_ln)
        x = self.first_conv(input_ln)
        interior_tensor_output.append(x)
        b = x.shape[0]
        c = x.shape[1]

        for idx in range(self.num_blocks):
            RR = self.RR
            x_rdy1 = x.clone()
            x_rdy2 = x.clone()
            interior_tensor_output.append(x_rdy1)
            interior_tensor_output.append(x_rdy2)
            linear_x = self.convs[idx](x_rdy1)
            rc = 4
            assert rc <= RR

            l1 = x.shape[-1]
            l2 = x.shape[-2]
            x_rdy2 = x_rdy2.reshape(b * c, 1, l2, l1)
            Legendre_x = F.conv2d(x_rdy2, self.filter_d / self.k / self.k,
                                  stride=self.n // self.k)  # shape: [b * c, self.modes, _, _]
            ll1 = Legendre_x.shape[-1]
            ll2 = Legendre_x.shape[-2]
            if self.recept == 0:
                print('remaining space: ', ll)

            Legendre_x = Legendre_x.reshape(b, c, self.modes, ll2, ll1)
            Legendre_x = Legendre_x.reshape(b, self.modes * c, ll2, ll1)

            Legendre_x2 = self.linearModes[idx](Legendre_x)

            Legendre_x2 = Legendre_x2.reshape(b, self.modes, c, ll2, ll1)
            Legendre_x2 = Legendre_x2.reshape(b * c, self.modes, ll2, ll1)

            Legendre_x = Legendre_x2 * 1
            Legendre_x = F.conv_transpose2d(Legendre_x, self.filter_r, stride=self.n // self.k)

            if rc == RR:
                Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:, :, RR:-RR, RR:-RR] + linear_x[:, :, :, :]
            else:
                Legendre_x = Legendre_x.reshape(b, c, l2, l1)[:, :, RR:-RR, RR:-RR] + linear_x[:, :, RR - rc:-RR + rc, RR - rc:-RR + rc]

            x_after = Legendre_x.clone()
            x = F.gelu(x_after)

        x = self.conv11(x)
        x = F.gelu(x)
        x = self.convout(x)

        x = x
        xx = x.clone()

        return xx,interior_tensor_output

if __name__ == '__main__':
    N = 16
    K = 2
    M = 6
    Params = {
        'n': N,
        'k': K,
        'm': M,
        'norm_factors': [1, 1, 1, 1], #[0.5, 0.5, 5, 10],  # [1, 1, 1, 1], #
        'init_weight': [math.sqrt(3), math.sqrt(Initialization_factors['({},{},{})'.format(N,K,M)]),
                        math.sqrt(6), math.sqrt(3), math.sqrt(6), math.sqrt(3)], # [1,1,1,1,1,1], #
        'if_ln': True  # False
    }
    n_layers = 4
    Activate = True
    if not Activate:
        Params['init_weight'][1] = Params['init_weight'][1] / math.sqrt(2)
        Params['init_weight'][2] = Params['init_weight'][2] / math.sqrt(2)
        Params['init_weight'][3] = Params['init_weight'][3] / math.sqrt(2)
        Params['init_weight'][4] = Params['init_weight'][4] / math.sqrt(2)

    network_name = 'initial_n{}N{}M{}K{}'.format(n_layers, Params['n'], Params['m'], Params['k'])

    k_x = 64
    k_y = 64

    grad = torch.zeros(128, 128)

    round = 300
    c = 1

    Nk = int(Params['n'] / Params['k'])
    for a in range(Nk):
        for b in range(Nk):
            grad_temp = torch.zeros(round, c, 128, 128)
            for i in range(round):
                network = NetComNS_InN_legendre(num_blocks=n_layers, Params=Params)
                input = Variable(torch.ones(1, c, 128, 128), requires_grad=True) * torch.randn(1) * 0.05
                #input = Variable(torch.randn(1, c, 128, 128), requires_grad=True)

                network.cuda()
                input = input.cuda()
                input.retain_grad()

                output, interiors = network(input)
                for interior in interiors:
                    interior.retain_grad()

                output_sum = torch.sum(output[:, :, k_x // Nk * Nk + a, k_y // Nk * Nk + b])

                network.zero_grad()
                output_sum.backward(retain_graph=True)

                grad_temp[i, :, :, :] = input.grad[0, :, :, :].cpu()

                print('{},{}'.format(a, b))

            grad_temp = torch.std(grad_temp,(0,1),unbiased=False)
            grad[:-a-1,:-b-1] = grad[:-a-1,:-b-1] + grad_temp[a:-1,b:-1]

    scio.savemat('receptives/{}_receptive'.format(network_name), mdict={"grad": grad.numpy()})

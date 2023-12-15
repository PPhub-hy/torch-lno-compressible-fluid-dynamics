import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import scipy.io as io

def train_iterative_ComNS_InN(network, batch_size,
                           epochs, max_ep, last_ep, optimizer,
                           dataset, train_gen, round,
                           print_frequency,
                           In_length):
    N = 128
    length_in = In_length
    network.train()
    iterations = 500

    losses_epoch = [0 for _ in range(round)]
    optimizer.zero_grad()
    for e in range(epochs):
        Lr = optimizer.param_groups[0]['lr']
        print(' ')
        print('*' * 20)
        print('epoch {}/{}:'.format(last_ep + e + 1, max_ep))
        print('Lr： {}'.format(Lr))
        last_time = time.time()
        losses = [0 for _ in range(round)]
        losses_relate = [0 for _ in range(round)]
        for i in range(iterations):
            data_input, data_output = next(train_gen)

            assert len(data_output) == round

            data_input = data_input.cuda()
            #data_input[:,1:,:,:] = 0
            for idx in range(len(data_output)):
                data_output[idx] = data_output[idx].cuda()
                #data_output[idx][:,1:,:,:] = 0

            loss_b = 0
            next_in = data_input
            for j in range(round):
                output = network(next_in)

                diff_norm = 0
                out_norm = 0
                for field in range(output.shape[1]):
                    diff = torch.norm(output[:, field, :, :].reshape(batch_size, - 1) \
                                      - data_output[j][:, field, :, :].reshape(batch_size, -1), 2, 1)
                    diff_norm = diff_norm + diff * network.norm_factors[field]
                    out_norm = out_norm + torch.norm(data_output[j][:, field, :, :].reshape(batch_size, -1), 2, 1) * \
                               network.norm_factors[field]

                loss_relate = torch.mean(diff_norm / out_norm)

                loss = torch.mean(diff_norm)
                loss_b += loss

                losses_relate[j] += loss_relate.item() / print_frequency

                losses[j] += loss.item() / print_frequency
                losses_epoch[j] += loss.item() / iterations

                if j + 1 <= round:
                    new_in = torch.zeros(batch_size, 4 * length_in, N, N).cuda()
                    home = 4 * (length_in - 1)
                    new_in[:, :home, :, :] = next_in[:, 4:, :, :]
                    new_in[:, -4:, :, :] = output
                    next_in = new_in

            loss_b.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5)

            optimizer.step()
            optimizer.zero_grad()

            if i % print_frequency == print_frequency - 1:
                print('iteration={}/{}'.format(i + 1, iterations))
                print('loss={}({})'.format(losses, losses_relate))
                print('time costs per iteration: {:.2f}'.format((time.time() - last_time) / print_frequency))
                last_time = time.time()
                losses = [0 for _ in range(round)]
                losses_relate = [0 for _ in range(round)]

        print('losses_epoch=', losses_epoch)
        losses_epoch = [0 for _ in range(round)]

import random
import numpy as np
import torch as T
nn = T.nn
F = nn.functional



class MLPNet(nn.Module):
    def __init__(self, ip_dim, op_dim, net_config, seed):
        print('Initialize MLP Network!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        net_arch = net_config['arch']
        activation = 'nn.' + net_config['activation']
        op_activation = 'nn.Identity' # net_config['output_activation']

        if net_arch > 0:
            layers = [nn.Linear(ip_dim, net_arch[0]), eval(activation)]
            for l in range(len(net_arch)-1):
                layers.extend([nn.Linear(net_arch[l], net_arch[l+1]), eval(activation)])
            if op_dim > 0:
                last_dim = net_arch[-1]
                layers.extend([nn.Linear(last_dim, op_dim), eval(op_activation)])
        else:
            raise 'No network arch!'

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)
import random
import numpy as np
import torch as T
from torch.nn.functional import layer_norm
nn = T.nn
F = nn.functional



class MLPNet(nn.Module):
    def __init__(self, ip_dim, op_dim, config, seed):
        print('Initialize MLP Network!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        layers = []
        net_arch = config['']

        if net_arch > 0:
            pass

        if op_dim > 0:
            pass

        self.net = nn.Sequential(*layers)
        pass

    def forward(self, x):
        return self.net(x)
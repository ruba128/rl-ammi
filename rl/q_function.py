import random
import numpy as np
import torch as T
import torch.nn as nn



class SoftQFunction(nn.Module):
    def __init__(self, seed):
        print('Initialize QFunction!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        pass

    def forward(self, o, a):
        pass
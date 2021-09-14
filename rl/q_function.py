import random
import numpy as np
import torch as T
nn = T.nn

from networks import MLPNet


class SoftQFunction(nn.Module):
    """
    Soft Q-Function
    """
    def __init__(self, seed):
        print('Initialize QFunction!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        pass

    def forward(self, o, a):
        pass
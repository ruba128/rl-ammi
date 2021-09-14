import random
import numpy as np
import torch as T
nn = T.nn

from networks import MLPNet


class StochasticPolicy(nn.Module):
    """
    Stochastic Gaussian Policy
    """
    def __init__(self, seed):
        print('Initialize Policy!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        pass

    def prob(self):
        pass

    def log_prob(self):
        pass

    def determinsitic(self):
        pass

    def forward(self, o):
        pass
import random
import numpy as np
import torch as T
import torch.nn as nn



class StochasticPolicy(nn.Module):
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
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F



class MLPNet(nn.Module):
    def __init__(self, seed):
        print('Initialize MLP Network!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        pass

    def forward(self):
        pass
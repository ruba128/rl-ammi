import random
import numpy as np
import torch as T
nn = T.nn

from networks import MLPNet


class SoftQFunction(nn.Module):
    """
    Soft Q-Function
    """
    def __init__(self, obs_dim, act_dim, net_config, seed):
        print('Initialize QFunction!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)

        self.critic1 = MLPNet(obs_dim + act_dim, 1, net_config, seed)
        self.critic2 = MLPNet(obs_dim + act_dim, 1, net_config, seed)
        
    def forward(self, o, a):

        #two critics
        Q_1 = self.critic1(T.cat([o, a], dim=1))
        Q_2 = self.critic2(T.cat([o, a], dim=1))


        return Q_1, Q_2


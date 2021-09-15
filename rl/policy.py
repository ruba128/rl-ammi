import random
import numpy as np
import torch as T
from torch.distributions.normal import Normal
nn = T.nn

from networks import MLPNet


class StochasticPolicy(nn.Module):
    """
    Stochastic Gaussian Policy
    """
    def __init__(self, ip_dim, op_dim, net_config, seed, max_action):
        print('Initialize Policy!')
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        net_arch = net_config['arch']
        self.reparam_noise = 1e-6
        self.model = MLPNet(ip_dim, op_dim, net_config, seed)
        self.mu = nn.Linear(net_arch[-1], num_actions)
        self.log_sigma = nn.Linear(net_arch[-1], num_actions)

    def prob(self, mu, sigma):
        probabilities = Normal(mu, sigma)
        actions = probabilities.rsample()
        return actions

    def log_prob(self,pi, mu, sigma):
        probabilities = Normal(mu, sigma)
        log_probs = probabilities.log_prob(pi)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

    def determinsitic(self,mu):
        return mu

    def get_mu_sigma(self, o):
        out_mlp = self.model(o)
        mu = self.mu(out_mlp)
        log_sigma = self.sigma(out_mlp)
        log_sigma = T.clamp(log_sigma, min=self.reparam_noise, max=1)
        sigma = T.exp(log_std)

        return mu, sigma


    def forward(self, o, deterministic= False):
        mu, sigma = self.get_mu_sigma(o)
        if deterministic == True:
            pi = self.deterministic(mu)
        else:
            pi = self.prob(o, mu, sigma)
        
        log_pi = self.log_prob(pi, mu, sigma) 
        action = T.tanh(pi)*T.tensor(self.max_action) ####max_action=env.action_space.high
        return action, log_pi

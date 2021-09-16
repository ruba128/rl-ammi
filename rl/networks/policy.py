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
    def __init__(self, ip_dim, op_dim, net_config, act_up_lim, act_low_lim, device, seed):
        print('Initialize Policy!')
        super().__init__() # To automatically use 'def forward'
        random.seed(seed), np.random.seed(seed), T.manual_seed(seed)
        net_arch = net_config['arch']
        self.reparam_noise = 1e-6
        
        # layers
        self.model = MLPNet(ip_dim, op_dim, net_config, seed)
        self.mu = nn.Linear(net_arch[-1], op_dim)
        self.log_sigma = nn.Linear(net_arch[-1], op_dim)

        # network parameters
        optimizer = 'nn.optim' + net_configs['critic']['network']['optimizer']
        lr = self.configs['critic']['network']['lr']
        self.optimizer = eval(optimizer)(self.parameters(), lr)

        # action space dimensions, scaling, and bias
        self.op_dim = op_dim
        self.act_scale = T.FloatTensor( 0.5 * (act_up_lim - act_low_lim) ).to(device)
        self.act_bias =  T.FloatTensor( 0.5 * (act_up_lim + act_low_lim) ).to(device)

    def prob(self, mu, sigma, reparameterize = True):
        probabilities = Normal(mu, sigma)

        # reparameterization trick
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()         
        return (actions * self.act_scale) + self.act_bias

    def log_prob(self, pi, mu, sigma):
        probabilities = Normal(mu, sigma)
        log_probs = probabilities.log_prob(pi)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return log_probs

    def determinsitic(self,mu):
        with T.no_grad(): 
            return (T.tanh(mu) * self.act_scale) + self.act_bias

    def get_mu_sigma(self, o):
        out_mlp = self.model(o)
        mu = self.mu(out_mlp)
        log_sigma = self.sigma(out_mlp)
        log_sigma = T.clamp(log_sigma, min=self.reparam_noise, max=1)
        sigma = T.exp(log_std)

        return mu, sigma


    def forward(self, o, deterministic= False, return_log_pi=True, reparameterize=True):
        mu, sigma = self.get_mu_sigma(o)
        log_pi = None

        if deterministic:
            pi = self.deterministic(mu)
        else:
            if return_log_pi:
                pi = self.prob(mu, sigma, reparameterize)
                log_pi = self.log_prob(pi, mu, sigma, reparameterize)
            else:
                pi = self.prob(mu, sigma, reparameterize)    
        
        return pi, log_pi

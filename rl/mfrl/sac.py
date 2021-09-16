import time
import wandb

import numpy as np
import torch as T
import torch.nn.functional as F

from mfrl.mfrl import MFRL
from policy import StochasticPolicy
from q_function import SoftQFunction



class ActorCritic: # Done
    """
    An entity contains both the actor (policy) that acts on the environment,
    and a critic (Q-function) that evaluate that state-action given a policy.
    """
    def __init__(self,
                 obs_dim, act_dim,
                 act_up_lim, act_low_lim,
                 configs, seed
                 ):

        # Initialize parameters
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.act_up_lim, self.act_low_lim = act_up_lim, act_low_lim
        self.config, self.seed = configs, seed
        self.device = configs['experiment']['device']

        self.actor, self.critic, self.critic_target = None, None, None
        self._build()


    def _build(self):
        self.actor = self._set_actor()
        self.critic = self._set_critic()
        self.critic_target = self._set_critic()
        # it is parameters will be updated using a weighted average
        for p in self.critic_target.parameters():
            p.require_grad = False


    def _set_actor(self):
        return StochasticPolicy(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            self.config, self.device, self.seed).to(self.device)


    def _set_critic(self):
        return SoftQFunction(
            self.obs_dim, self.act_dim,
            self.config, self.seed).to(self.device)



class SAC(MFRL):
    """
    Algorithm: Soft Actor-Critic (Off-policy, Model-free)

        01. Input: θ1, θ2, φ                                    > Initial parameters
        02. ¯θ1 ← θ1, ¯θ2 ← θ2                                  > Initialize target network weights
        03. D ← ∅                                               > Initialize an empty replay pool
        04.    for each iteration do
        05.       for each environment step do
        06.          at ∼ πφ(at|st)                             > Sample action from the policy
        07.          st+1 ∼ p(st+1|st, at)                      > Sample transition from the environment
        08.          D ← D ∪ {(st, at, r(st, at), st+1)}        > Store the transition in the replay pool
        09.       end for
        10.       for each gradient step do
        11.          θi ← θi − λ_Q ˆ∇θi J_Q(θi) for i ∈ {1, 2}  > Update the Q-function parameters
        12.          φ ← φ − λ_π ˆ∇φ J_π(φ)                     > Update policy weights
        13.          α ← α − λ ˆ∇α J(α)                         > Adjust temperature
        14.          ¯θi ← τ θi + (1 − τ) + ¯θi for i ∈ {1, 2}  > Update target network weights
        15.       end for
        16.    end for
        17. Output: θ1, θ2, φ                                   > Optimized parameters
  
    """
    def __init__(self, configs, seed):
        super(SAC, self).__init__(configs, seed)
        print('Initialize SAC Algorithm!')
        self.configs = configs
        self.seed = seed
        self._biuld()


    def _build(self):
        super(SAC, self)._build()
        self._build_sac()


    def _build_sac(self):
        self._set_actor_critic()
        self._set_alpha()


    def _set_actor_critic(self):
        self.ator_critic = ActorCritic(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            self.configs, self.seed)


    def _set_alpha(self):
        if self.configs['actor']['automatic_entropy']:
            # Learned Temprature
            device = self.configs['experiment']['device']
            optimizer = 'nn.optim' + self.configs['actor']['network']['optimizer']
            lr = self.configs['actor']['network']['lr']
            target_entropy = self.configs['actor']['target_entropy']

            self.target_entropy = (
                - 1.0 * T.prod(
                    T.tensor(self.train_env.action_space.shape).to(device)
                ).item() if target_entropy == 'auto' else target_entropy
            )

            self.log_alpha = T.zeros(1, require_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = eval(optimizer)([self.log_alpha], lr)
        else:
            # Fixed Temprature
            self.alpha = self.configs['actor']['alpha']


    def learn(self):
        N = self.configs['algorithm']['learning']['epochs']
        NT = self.configs['algorithm']['learning']['epoch_steps']
        Ni = self.configs['algorithm']['learning']['init_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        G = self.configs['algorithm']['learning']['grad_AC_steps_steps']

        batch_size = self.configs['data']['batch_size']

        o, Z, el, t = self.env.reset(), 0, 0, 0
        oldJs = [0, 0, 0]
        JQList, JAlphaList, JPiList = [], [], []
        logs = dict()

        start_time_real = time.time()
        for n in range(1, N+1):
            nt, x = 0, (n * NT) / NT
            learn_start_real = time.time()
            while nt <= NT:
                # Interaction steps
                for e in range(1, E+1):
                    o, Z, el, t = self.internact(self.ator_critic.actor,
                                                 n, Ni,
                                                 o, Z, el, t)

                # Taking gradient steps after exploration
                if n > Ni:
                    for g in range(1, G+1):
                        batch = self.replay_buffer.sample(batch_size)
                        Jq, Jalpha, Jpi = self.trainAC(g, batch, oldJs)
                        oldJs = [Jq, Jalpha, Jpi]
                        JQList.append(Jq)
                        JAlphaList.append(Jalpha)
                        JPiList.append(Jpi)
                else:
                    JQList.append(0)
                    JAlphaList.append(0)
                    JPiList.append(0)

                nt += E
            logs['time/training'] = time.time() - learn_start_real
            logs['training/losses/Jq'] = np.mean(JQList)
            logs['training/losses/Jalpha'] = np.mean(JAlphaList)
            logs['training/losses/Jpi'] = np.mean(JPiList)

            eval_start_real = time.time()
            self.evaluate(self.ator_critic.actor, x)
            logs['time/evaluation'] = time.time() - eval_start_real
            
        logs['time/total'] = time.time() - start_time_real


    def trainAC(self, g, batch, oldJs):
        AUI = self.configs['algorithm']['learning']['alpha_update_interval']
        PUI = self.configs['algorithm']['learning']['policy_update_interval']
        TUI = self.configs['algorithm']['learning']['target_update_interval']

        Jq = self.updateQ(batch)

        if g % AUI == 0:
            Jalpha = self.updateAlpha(batch)
        else:
            Jpi = oldJs[1]

        if g % PUI == 0:
            Jpi = self.updatePi(batch)
        else:
            Jpi = oldJs[2]
        
        if g % TUI == 0:
            self.updateTarget()

        return Jq, Jalpha, Jpi


    def updateQ(self, batch):
        """"
        JQ(θ) = E(st,at)∼D[ 0.5 ( Qθ(st, at)
                            − r(st, at)
                            + γ Est+1∼D[ Eat+1~πφ(at+1|st+1)[ Qθ¯(st+1, at+1)
                                                − α log(πφ(at+1|st+1)) ] ] ]
        """
       
        O = batch['observations']
        A = batch['actions']
        R = batch['rewards']
        O_next = batch['observations_next']
        D = batch['terminals']

        gamma = self.configs['critic']['gamma']

        # Calculate two Q-functions
        Qs = self.ator_critic.critic(O, A)

        # Bellman backup for Qs
        with T.no_grad():
            pi, log_pi = self.ator_critic.actor(O, reparam=True, return_log_pi=True)
            A_next = pi
            Qs_targ = T.cat(self.ator_critic.critic(O_next, A_next), dim=1)
            min_Q_targ, _ = T.min(Qs_targ, dim=1, keep_dim=True)
            Qs_backup = R + (1-D) * gamma * (min_Q_targ - self.alpha * log_pi)

        # MSE loss
        Jq = 0.5 * sum([F.mse_loss(Q, Qs_backup) for Q in Qs])

        # Gradient Descent
        self.ator_critic.critic.optimizer.zero_grad()
        Jq.backward()
        self.ator_critic.critic.optimizer.step()
        
        return Jq


    def updateAlpha(self, batch):
        """
        
        αt* = arg min_αt Eat∼πt∗[ −αt log( πt*(at|st; αt) ) − αt H¯

        """
        if self.configs['actor']['automatic_entropy']:
            # Learned Temprature
            O = batch['observations']

            with T.no_grad():
                _, log_pi = self.ator_critic.actor(O, reparameterize=True, return_log_pi=True)
            Jalpha = - (self.log_alpha * (log_pi + self.target_entropy)).item()

            # Gradient Descent
            self.alpha_optimizer.zero_grad()
            Jalpha.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

            return Jalpha
        else:
            # Fixed Temprature
            return 0


    def updatePi(self, batch):
        """
        Jπ(φ) = Est∼D[ Eat∼πφ[α log (πφ(at|st)) − Qθ(st, at)] ]
        """

        O = batch['observations']

        # Policy Evaluation
        pi, log_pi = self.ator_critic.actor(O, reparam=True, return_log_pi=True)
        Qs_pi = T.cat(self.ator_critic.critic(O, pi), dim=1)
        min_Q_pi, _ = T.min(Qs_pi, dim=1, keep_dim=True)

        # Policy Improvement
        Jpi = (self.alpha * log_pi - min_Q_pi).mean()

        # Gradient Ascent
        self.ator_critic.actor.optimizer.zero_grad()
        Jpi.backward()
        self.ator_critic.actor.optimizer.step()
        
        return Jpi


    def updateTarget(self):
        tau = self.configs['critic']['tau']
        with T.no_grad():
            for p, p_targ in zip(self.ator_critic.critic.parameters(),
                                 self.ator_critic.critic_target.parameters()):
                p_targ.data.copy_(tau * p.data + (1-tau) * p_targ.data)


# Report:
#   Explain the problem, algorithm, do anaysis, and show results.
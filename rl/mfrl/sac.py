import time
import wandb

import numpy as np
import torch as T
import torch.nn.functional as F

from mfrl.mfrl_ import MFRL
from networks.policy_ import StochasticPolicy
from networks.q_function_ import SoftQFunction



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
        print('Initialize AC!')
        # Initialize parameters
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.act_up_lim, self.act_low_lim = act_up_lim, act_low_lim
        self.configs, self.seed = configs, seed
        self.device = configs['experiment']['device']

        self.actor, self.critic, self.critic_target = None, None, None
        self._build()


    def _build(self):
        self.actor = self._set_actor()
        self.critic = self._set_critic()
        self.critic_target = self._set_critic()
        # it is parameters will be updated using a weighted average
        for p in self.critic_target.parameters():
            p.requires_grad = False


    def _set_actor(self):
        net_configs = self.configs['actor']['network']
        return StochasticPolicy(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            net_configs, self.device, self.seed).to(self.device)


    def _set_critic(self):
        net_configs = self.configs['actor']['network']
        return SoftQFunction(
            self.obs_dim, self.act_dim,
            net_configs, self.seed).to(self.device)



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
    def __init__(self, exp_prefix, configs, seed):
        super(SAC, self).__init__(exp_prefix, configs, seed)
        print('Initialize SAC Algorithm!')
        self.configs = configs
        self.seed = seed
        self._build()


    def _build(self):
        super(SAC, self)._build()
        self._build_sac()


    def _build_sac(self):
        self._set_actor_critic()
        self._set_alpha()


    def _set_actor_critic(self):
        self.actor_critic = ActorCritic(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            self.configs, self.seed)


    def _set_alpha(self):
        if self.configs['actor']['automatic_entropy']:
            # Learned Temprature
            device = self.configs['experiment']['device']
            optimizer = 'T.optim.' + self.configs['actor']['network']['optimizer']
            lr = self.configs['actor']['network']['lr']
            target_entropy = self.configs['actor']['target_entropy']

            self.target_entropy = (
                - 1.0 * T.prod(
                    T.tensor(self.learn_env.action_space.shape).to(device)
                ).item() if target_entropy == 'auto' else target_entropy
            )

            self.log_alpha = T.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = eval(optimizer)([self.log_alpha], lr)
        else:
            # Fixed Temprature
            self.alpha = self.configs['actor']['alpha']


    def learn(self):
        N = 20 #self.configs['algorithm']['learning']['epochs']
        NT = 500 #self.configs['algorithm']['learning']['epoch_steps']
        Ni = 5 #self.configs['algorithm']['learning']['init_epochs']

        E = self.configs['algorithm']['learning']['env_steps']
        G = self.configs['algorithm']['learning']['grad_AC_steps']

        batch_size = self.configs['data']['batch_size']

        # o, Z, el, t = self.env.reset(), 0, 0, 0
        o, Z, el, t = self.initialize_learning(NT, Ni)
        oldJs = [0, 0, 0]
        JQList, JAlphaList, JPiList = [], [], []
        logs = dict()
        
        start_time_real = time.time()
        for n in range(Ni+1, N+1):
            print('\n[ Learning ]')
            nt, x = 0, (n * NT) / NT
            learn_start_real = time.time()
            while nt <= NT:
                # Interaction steps
                for e in range(1, E+1):
                    o, Z, el, t = self.internact(self.actor_critic.actor,
                                                 o, Z, el, t)

                # Taking gradient steps after exploration
                if n > Ni:
                    for g in range(1, G+1):
                        batch = self.replay_buffer.sample_batch(batch_size)
                        Jq, Jalpha, Jpi = self.trainAC(g, batch, oldJs)
                        oldJs = [Jq, Jalpha, Jpi]
                        JQList.append(Jq.item())
                        JAlphaList.append(Jalpha.item())
                        JPiList.append(Jpi.item())
                else:
                    JQList.append(0)
                    JAlphaList.append(0)
                    JPiList.append(0)

                nt += E
            logs['time/training'] = round(time.time() - learn_start_real, 2)
            logs['training/objectives/sac/Jq'] = np.mean(JQList)
            logs['training/objectives/sac/Jalpha'] = np.mean(JAlphaList)
            logs['training/objectives/sac/Jpi'] = np.mean(JPiList)

            eval_start_real = time.time()
            AvgEZ, AvgES, AvgEL = self.evaluate(self.actor_critic.actor, x)
            logs['time/evaluation'] = round(time.time() - eval_start_real, 2)
            logs['evaluation/avg_episodic_return'] = AvgEZ
            # logs['evaluation/avg_episodic_score'] = AvgES
            logs['evaluation/avg_episodic_length'] = AvgEL
            
            logs['time/total'] = round(time.time() - start_time_real, 2)

            # Printing logs
            if self.configs['experiment']['print_logs']:
                print('=' * 80)
                print(f'Epoch {n}')
                for k, v in logs.items():
                    print(f'{k}: {v}')

            # # WandB
            # if self.configs['experiment']['WandB']:
            #     wandb.log(logs)


    def trainAC(self, g, batch, oldJs):
        AUI = self.configs['algorithm']['learning']['alpha_update_interval']
        PUI = self.configs['algorithm']['learning']['policy_update_interval']
        TUI = self.configs['algorithm']['learning']['target_update_interval']

        Jq = self.updateQ(batch)
        Jalpha = self.updateAlpha(batch) if (g % AUI == 0) else oldJs[1]
        Jpi = self.updatePi(batch) if (g % PUI == 0) else oldJs[2]
        if g % TUI == 0: self.updateTarget()

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
        Qs = self.actor_critic.critic(O, A)

        # Bellman backup for Qs
        with T.no_grad():
            pi, log_pi = self.actor_critic.actor(O, reparameterize=True, return_log_pi=True)
            A_next = pi
            Qs_targ = T.cat(self.actor_critic.critic(O_next, A_next), dim=1)
            min_Q_targ, _ = T.min(Qs_targ, dim=1, keepdim=True)
            # print('R: ', R.shape)
            # print('D: ', D.shape)
            # print('min_Q_targ: ', min_Q_targ.shape)
            # print('self.alpha: ', self.alpha)
            # print('log_pi: ', log_pi.shape)
            Qs_backup = R + gamma * (1-D) * (min_Q_targ - self.alpha * log_pi)

        # MSE loss
        Jq = 0.5 * sum([F.mse_loss(Q, Qs_backup) for Q in Qs])

        # Gradient Descent
        self.actor_critic.critic.optimizer.zero_grad()
        Jq.backward()
        self.actor_critic.critic.optimizer.step()
        
        return Jq


    def updateAlpha(self, batch):
        """
        
        αt* = arg min_αt Eat∼πt∗[ −αt log( πt*(at|st; αt) ) − αt H¯

        """
        if self.configs['actor']['automatic_entropy']:
            # Learned Temprature
            O = batch['observations']

            with T.no_grad():
                _, log_pi = self.actor_critic.actor(O, reparameterize=True, return_log_pi=True)
            # print('self.log_alpha: ', self.log_alpha)
            # print('log_pi: ', log_pi)
            # print('self.target_entropy: ', self.target_entropy)
            Jalpha = - (self.log_alpha * (log_pi + self.target_entropy)).mean()

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
        pi, log_pi = self.actor_critic.actor(O, reparameterize=True, return_log_pi=True)
        Qs_pi = T.cat(self.actor_critic.critic(O, pi), dim=1)
        min_Q_pi, _ = T.min(Qs_pi, dim=1, keepdim=True)

        # Policy Improvement
        Jpi = (self.alpha * log_pi - min_Q_pi).mean()

        # Gradient Ascent
        self.actor_critic.actor.optimizer.zero_grad()
        Jpi.backward()
        self.actor_critic.actor.optimizer.step()
        
        return Jpi


    def updateTarget(self):
        tau = self.configs['critic']['tau']
        with T.no_grad():
            for p, p_targ in zip(self.actor_critic.critic.parameters(),
                                 self.actor_critic.critic_target.parameters()):
                p_targ.data.copy_(tau * p.data + (1-tau) * p_targ.data)


# Report:
#   Explain the problem, algorithm, do anaysis, and show results.
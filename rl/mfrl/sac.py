import torch as T

from mfrl import MFRL
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
                 config
                 ):

        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.act_up_lim, self.act_low_lim = act_up_lim, act_low_lim
        self.config, self.device = config, config['experiment']['device']

        self.actor, self.critic, self.critic_target = None, None, None
        self._biuld()

    def _biuld(self):
        self.actor = self._set_actor()
        self.critic = self._set_critic()
        self.critic_target = self._set_critic()
        for p in self.critic_target.parameters():
            p.require_grad = False

    def _set_actor(self):
        return StochasticPolicy(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            self.config
        ).to(self.device)

    def _set_critic(self):
        return SoftQFunction(
            self.obs_dim, self.act_dim,
            self.config
        ).to(self.device)


class SAC(MFRL):
    """
    Algorithm: Soft Actor-Critic (Off-policy, Model-free)
        01. Input: θ1, θ2, φ                                    . Initial parameters
        02. ¯θ1 ← θ1, ¯θ2 ← θ2                                  . Initialize target network weights
        03. D ← ∅                                               . Initialize an empty replay pool
        04.    for each iteration do
        05.       for each environment step do
        06.          at ∼ πφ(at|st)                             . Sample action from the policy
        07.          st+1 ∼ p(st+1|st, at)                      . Sample transition from the environment
        08.          D ← D ∪ {(st, at, r(st, at), st+1)}        . Store the transition in the replay pool
        09.       end for
        10.       for each gradient step do
        11.          θi ← θi − λ_Q ˆ∇θi J_Q(θi) for i ∈ {1, 2} . Update the Q-function parameters
        12.          φ ← φ − λ_π ˆ∇φ J_π(φ)                    . Update policy weights
        13.          α ← α − λ ˆ∇α J(α)                        . Adjust temperature
        14.          ¯θi ← τ θi + (1 − τ) + ¯θi for i ∈ {1, 2}  . Update target network weights
        15.       end for
        16.    end for
        17. Output: θ1, θ2, φ                                   . Optimized parameters
    """
    def __init__(self):
        super(SAC, self).__init__()
        print('Initialize SAC Algorithm!')
        pass


    def _biuld(self):
        super(SAC, self)._biuld()
        self._build_sac()

    def _biuld_sac(self):
        self._set_actor_critic()
        self._set_alpha()

    def _set_actor_critic(self):
        self.ator_critic = ActorCritic(
            self.obs_dim, self.act_dim,
            self.act_up_lim, self.act_low_lim,
            self.config
        )

    def _set_alpha(self):
        pass

    def learn(self):
        pass

    def updateAC(self, batch):
        pass

    def updateQ(self, batch):
        pass

    def updateAlpha(self, batch):
        pass

    def updatePi(self, batch):
        pass

    def updateTarget(self):
        pass


# Report:
#   Explain the problem, algorithm, do anaysis, and show results.
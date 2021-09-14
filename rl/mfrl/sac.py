import torch as T

from mfrl import MFRL
from policy import StochasticPolicy
from q_function import SoftQFunction


class ActorCritic:
    """
    An entity contains both the actor (policy) that acts on the environment,
    and a critic (Q-function) that evaluate that state-action given apolicy
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
    def __init__(self):
        super(SAC, self).__init__()
        print('Initialize SAC!')
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


    
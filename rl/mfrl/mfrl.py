import gym

from buffer import ReplayBuffer


class MFRL:
    def __init__(self, config):
        print('Initialize MFRL!')
        self.env_name = config['environment']['name']
        self.env_horizon = config['environment']['env_horizon']
        self.max_size = config['environment']['model_buffer_size']
        self.seed = config['experiment']['seed']
        self.num_test_eps = config['evaluation']['eval_episodes']
        self._build()

    def _build(self):
        self._set_env()
        self._set_replay_buffer()

    def _set_env(self):
        env_name = self.env_name
        # Inintialize Learning environment:
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        # Ininialize Evaluation environment:
        self.eval_env = gym.make(env_name)

    def _set_replay_buffer(self):
        self.buffer =  ReplayBuffer(self.obs_dim, self.act_dim, self.max_size, self.seed, self.device)

    def initialize_learning(self, init_epochs):
        for i in range(init_epochs):
            pass
        self.state = self.env.reset()

    def interact(self, action):
        '''
        act on the enviroment and return next state, and reward

        Parameter

        Returns
        '''
        next_state, reward, d , info = self.env.step(action)
        self.state = os
        self.buffer.store_transition(self.state, action, reward, next_state, d)


    def evaluate(self, policy, eval_episodes):
        for j in range(eval_episodes):
            o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r 
                ep_len += 1
            # TODO: there should be some kind of logging here
            #logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

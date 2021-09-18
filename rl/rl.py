import gym


from buffer import ReplayBuffer


# class RL:
#     def __init__(self, configs, seed):
#         print('Initialize RL!')
#         self.configs = configs
#         self.seed = seed


#     def _build(self):
#         self._set_env()
#         self._set_replay_buffer()


#     def _set_env(self):
#         env_name = self.configs['environment']['name']
#         evaluate = self.configs['algorithm']['evaluation']
        
#         # Inintialize Learning environment
#         self.learn_env = gym.make(env_name)
#         self._seed_env(self.learn_env, self.seed)

#         if evaluate:
#             # Ininialize Evaluation environment
#             self.eval_env = gym.make(env_name)
#             self._seed_env(self.eval_env, self.seed)

#         # Spaces dimensions
#         self.obs_dim = self.learn_env.observation_space.shape[0]
#         self.act_dim = self.learn_env.action_space.shape[0]
#         self.act_up_lim = self.learn_env.action_space.high
#         self.act_low_lim = self.learn_env.action_space.low
    

#     def _seed_env(self, env, seed):
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)


#     def _set_replay_buffer(self):
#         max_size = self.configs['data']['buffer_size']
#         device = self.configs['experiment']['device']
#         self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim,
#                                           max_size, self.seed, device)


#     def initialize_learning(self, NT, Ni):
#         env = self.learn_env
#         Denv = self.replay_buffer
#         max_el = self.configs['environment']['horizon']

#         o, Z, el, t = env.reset(), 0, 0, 0

#         if Ni < 1: return o, Z, el, t
        
#         print(f'[ Initial exploaration ] Starting')
#         for ni in range(1, Ni+1):
#             print(f'[ Initial exploaration ] Epoch {ni}')
#             nt = 0
#             while nt <= NT:
#                 # Random actions
#                 a = env.action_space.sample()
#                 o_next, r, d, info = env.step(a)
#                 d = True if el == max_el else d

#                 Denv.store_transition(o, a, r, o_next, d)

#                 o = o_next
#                 Z += r
#                 el +=1
#                 t +=1
                
#                 if d or (el == max_el):
#                     o, Z, el = env.reset(), 0, 0
                
#                 nt += 1

#         return o, Z, el, t



#     def internact(self, policy, o, Z, el, t):
#         env = self.learn_env
#         Denv = self.replay_buffer
#         max_el = self.configs['environment']['horizon']

#         # Use policy
#         a, _ = policy.step_np(o)

#         o_next, r, d, _ = env.step(a)
#         d = True if el == max_el else d

#         Denv.store_transition(o, a, r, o_next, d)

#         o = o_next
#         Z += r
#         el +=1
#         t +=1
        
#         if d or (el == max_el):
#             o, Z, el = env.reset(), 0, 0

#         # TODO: loggers

#         return o, Z, el, t


#     def evaluate(self, policy, x):
#         evaluate = self.configs['algorithm']['evaluation']
#         if evaluate:
#             print('[ Evaluation ]')
#             EE = self.configs['algorithm']['evaluation']['eval_episodes']
#             env = self.eval_env
#             max_el = self.configs['environment']['horizon']
#             EZ = 0 # Evaluation episodic return
#             # ES = 0 # Evaluation episodic score
#             EL = 0 # Evaluation episodic 

#             for ee in range(1, EE+1):
#                 o, d, Z, S, el = env.reset(), False, 0, 0, 0
#                 while not(d or (el == max_el)):
#                     # Take deterministic actions at evaluation time
#                     pi, _ = policy(o, deterministic=True)
#                     a = pi.cpu().numpy()
#                     o, r, d, info = env.step(a)
#                     Z += r
#                     # S += info['score']
#                     el += 1
#                 EZ += Z
#                 # ES += S
#                 EL += el
#                 # TODO: loggers

#             AvgEZ = EZ / EE
#             AvgES = 0 # ES / EE
#             AvgEL = EL / EE
#         return AvgEZ, AvgES, AvgEL
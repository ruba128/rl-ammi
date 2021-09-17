

configurations = {
    'environment': {
            'name': 'Hopper-v3',
            'type': 'gym-mujoco',
            'state_space': 'continuous',
            'action_space': 'continuous',
            'horizon': 1e3,
        },

    'algorithm': {
        'alg_name': 'SAC',
        'learning': {
            'epochs': 1000, # N epochs
            'epoch_steps': 1000, # NT steps/epoch
            'init_epochs': 10, # Ni epochs
            # 'expl_epochs': 10, # Nx epochs

            'env_steps' : 50, # E: interact E times then train
            'grad_AC_steps': 50, # ACG: ac grad
            
            'policy_update_interval': 1,
            'alpha_update_interval': 1,
            'target_update_interval': 1,
                    },

        'evaluation': {
            'evaluate': True,
            'eval_deterministic': True,
            'eval_freq': 1, # Evaluate every 'eval_freq' epochs --> Ef
            'eval_episodes': 5, # Test policy for 'eval_episodes' times --> EE
            'eval_render_mode': None,
        }
    },

    'actor': {
        'type': 'gaussianpolicy',
        'action_noise': None,
        'alpha': 0.2, # Temprature/Entropy #@#
        'automatic_entropy': True,
        'target_entropy': 'auto',
        'network': {
            'arch': [256,256],
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4
        }
    },

    'critic': {
        'type': 'sofQ',
        'number': 2,
        'gamma': 0.99,
        'tau': 5e-3,
        'network': {
            'arch': [256,256],
            'activation': 'ReLU',
            'output_activation': 'nn.Identity',
            'optimizer': "Adam",
            'lr': 3e-4
        }
    },

    'data': {
        'buffer_type': 'simple',
        'buffer_size': int(1e6),
        'batch_size': 256
    },

    'experiment': {
        'name': 'seed1',
        'seed': 1,
        'verbose': 0,
        'device': "cpu",
        'wandb_log': True,
        'print_logs': True,
        'logdir': 'tmp/sac',
        'capture_video': True
    }
    
}

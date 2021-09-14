import sys
import argparse
import importlib
import time
import datetime
import random

import numpy as np
import torch as T
import wandb

from mfrl.sac import SAC

def main(configs):
    print('Start Soft Actor-Critic experiment...')
    print('\n')
    env_name = configs['experiment']['env_name']
    env_type = 'mujoco'

    group_name = f"gym-{env_type}-{env_name}"
    now = datetime.datetime.now()
    exp_prefix = f"{group_name}-{now.year}/{now.month}/{now.day}-->{now.hour}:{now.minute}:{now.second}"

    print('=' * 50)
    print(f'Starting new experiment: {env_type}-{env_name}')
    print('=' * 50)
    
    if configs['experiment']['WandB']:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='sac-ammi',
            # project='rand',
            config=configs
        )

    # experiment = SAC(configs)

    # experiment.learn()
    # experiment.evaluate()

    print('\n')
    print('...End Soft Actor-Critic experiment')
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', type=str)
    parser.add_argument('-cfg_path', type=str)

    args = parser.parse_args()

    sys.path.append("./config")
    config = importlib.import_module(args.cfg)
    print('configurations: ', config.configurations)

    main(config.configurations)
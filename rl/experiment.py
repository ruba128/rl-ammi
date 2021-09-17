import sys
import argparse
import importlib
import time
import datetime
import random

import numpy as np
import torch as T
# import wandb

from mfrl.sac import SAC

def main(configs, seed):
    print('\n')
    alg_name = configs['algorithm']['alg_name']
    env_name = configs['environment']['env_name']
    env_type = configs['environment']['env_type']

    group_name = f"{env_type}-{env_name}"
    now = datetime.datetime.now()
    exp_prefix = f"{group_name}-{seed}-{now.year}/{now.month}/{now.day}-->{now.hour}:{now.minute}:{now.second}"

    print('=' * 50)
    print(f'Starting a new experiment:')
    print(f"\t Algorithm:   {alg_name}")
    print(f"\t Environment: {env_name}")
    print(f"\t Random seed: {seed}")
    print('=' * 50)
    
    # if configs['experiment']['WandB']:
    #     wandb.init(
    #         name=exp_prefix,
    #         group=group_name,
    #         # project='sac-ammi',
    #         project='rand',
    #         config=configs
    #     )

    agent = SAC(configs)
    agent.learn()
    # agent.evaluate()

    print('\n')
    print('End of the experiment.')
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', type=str)
    # parser.add_argument('-cfg_path', type=str)
    parser.add_argument('-seed', type=str)

    args = parser.parse_args()

    sys.path.append("./configs")
    config = importlib.import_module(args.cfg)
    seed = args.seed
    print('configurations: ', config.configurations)

    main(config.configurations, seed)
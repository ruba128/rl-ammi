# RL-AMMI
## AMMI, Deep RL, Fall 2021: RL Implementation for Continuous Control Tasks


## Course and Project details
This Deep RL course was taught at **The African Master's in Machine Intelligence** [AMMI](https://aimsammi.org/) in Fall 2021. It was instructed by researchers at [DeepMind](https://deepmind.com/): *[Bilal Piot](https://scholar.google.com/citations?user=fqxNUREAAAAJ)*, *[Corentin Tallec](https://scholar.google.com/citations?user=OPKX4GgLCxIC)* and *[Florian Strub](https://scholar.google.com/citations?user=zxO5kccAAAAJ)*. This project is the coursework of Deep RL where we **Catalyst Agents** team trying to re-implement RL algorithm(s) for continuous control tasks. We chose three types of environments: easy, medium, and hard to run the algorithm(s). The course project meant to submit only one algorithm, but we plan to continue working on this repo making it an open project by this team of student from AMMI. This is why we're trying to make a modular repo to ease the re-implementation of future algorithms.



## Algorithm:
Algorithm we re-implementing/plannning to re-implement:
1. Soft Actor-Critic (SAC) [Paper](https://arxiv.org/abs/1812.05905) (Now)

2. Model-Based Policy Optimization (MBPO) [Paper](https://arxiv.org/abs/1812.05905) (Next; Future work)

3. Model Predictive Control-Soft Actor Critic (MPC-SAC) [Paper](https://ieeexplore.ieee.org/document/9429677) (Next; Future work)

4. Model Predictive Actor-Critic (MoPAC) [Paper](https://arxiv.org/abs/2103.13842) (Next; Future work)



## How to use this code
### Installation
#### Ubuntu 20.04

Move into `rl-ammi` directory, and then run the following:

```
conda create -n rl-ammi python=3.8

pip install -e .

pip install numpy

pip install torch

pip install wandb

pip install gym
```

If you want to run MuJoCo Locomotion tasks, and ShadowHand, you should install [MuJoCo](http://www.mujoco.org/) first (it's open sourced until 31th Oct), and then install [mujoco-py](https://github.com/openai/mujoco-py):
```
sudo apt-get install ffmpeg

pip install -U 'mujoco-py<2.1,>=2.0'
```

If you are using A local GPU of Nvidia and want to record MuJoCo environments [issue link](https://github.com/openai/mujoco-py/issues/187#issuecomment-384905400), run:
```
unset LD_PRELOAD
```

#### MacOS

Move into `rl-ammi` directory, and then run the following:

```
conda create -n rl-ammi python=3.8

pip install -e .

pip install numpy

pip install torch

pip install wandb

pip install gym
```

If you want to run MuJoCo Locomotion tasks, and ShadowHand, you should install [MuJoCo](http://www.mujoco.org/) first (it's open sourced until 31th Oct), and then install [mujoco-py](https://github.com/openai/mujoco-py):
```
brew install ffmpeg

pip install -U 'mujoco-py<2.1,>=2.0'
```

If you are using A local GPU of Nvidia and want to record MuJoCo environments [issue link](https://github.com/openai/mujoco-py/issues/187#issuecomment-384905400), run:
```
unset LD_PRELOAD
```



### Run an experiment

Move into `rl-ammi/` directory, and then:

```
python experiment.py -cfg <cfg_file-.py> -seed <int>
```
for example:

```
python experiment.py -cfg sac_hopper -seed 1
```

### Evaluate an Agent
To evaluate a saved policy model, run the following command:
```
python evaluate_agent.py -env <env_name> -alg <alg_name> -seed <int> -EE <int>
```
for example:

```
python evaluate_agent.py -env Walker2d-v2 -alg SAC -seed 1 -EE 5
```


## Experiments and Results
### Classic Control

### MuJoCo Locomotion

### ShadowHand



## Catalyst Agents Team, Group 2
(first name alphabetical order)
- [MohammedElfatih Salah](https://github.com/mohammedElfatihSalah)
- Rami Ahmed
- [Ruba Mutasim](https://github.com/ruba128)
- [Wafaa Mohammed](https://github.com/Wafaa014)



## Accknowledgement
This repo was inspired by many great repos, mostly the following ones:
- [SpinningUp](https://github.com/openai/spinningup)
- [Stabel Baselines](https://github.com/hill-a/stable-baselines)
- [RLKit](https://github.com/rail-berkeley/rlkit)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Youtube-Code-Repository](https://github.com/philtabor/Youtube-Code-Repository)



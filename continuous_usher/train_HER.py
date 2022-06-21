import numpy as np
import gym 
import os, sys 
from mpi4py import MPI 
import pdb 
from HER.arguments import get_args 

import random
import torch

sys.path.append('./pybullet_env')

import pickle
"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

LOGGING = True


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def launch(args):
    from env_maker import make_env
    env = make_env(args)

    try: 
        env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    except: 
        pass
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # create the ddpg agent to interact with the environment 

    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()
    return ddpg_trainer

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()


    from HER.rl_modules.usher_agent_high_dim import ddpg_agent
    suffix = ""

    agent = launch(args)


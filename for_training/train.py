import gymnasium as gym
import numpy as np
from ballz_environment import *
from ai_trainer import *
import argparse as ap

parser = ap.ArgumentParser(prog="Train Ballz Agent")

parser.add_argument("-n", "--n-envs", choices=[str(i) for i in range(1,25)], required=True, help="Number of Ballz Environments to simultaneously train.")

args = parser.parse_args()

gym.register("gymnasium_env/Ballz-v0", entry_point=BallzEnv, max_episode_steps=1000)
#env = gym.make("gymnasium_env/Ballz-v0", width=700, height=950, nblocks=7, render_mode='human')
import os

def make_env(width=700, height=950, nblocks=7, render_mode=None):
    def _thunk():
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""

        env = gym.make("gymnasium_env/Ballz-v0", width=width, height=height, nblocks=nblocks, render_mode=render_mode)
        return env
    return _thunk

def cpu_worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue, autoreset_mode, semaphore):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Then use Gymnasium's normal worker implementation.
    gym.vector.async_vector_env._async_worker(
        index,
        env_fn,
        pipe,
        parent_pipe,
        shared_memory,
        error_queue,
        autoreset_mode,
        semaphore
    )

# def make_env():
#     return BallzEnv(width=700, height=950, nblocks=7, render_mode=None)
env = gym.vector.AsyncVectorEnv(
    [make_env() for _ in range(int(args.n_envs))],
    worker=cpu_worker
)

agent = BallzAgent(env, alpha=3.5e-4, gamma=0.9, epsilon=0.2)
rewards = agent.train(2000, live_plot=False, ppo=False, mc=False)

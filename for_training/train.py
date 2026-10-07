import gymnasium as gym
import numpy as np
from ai_trainer import *

gym.register("gymnasium_env/Ballz-v0", entry_point=BallzEnv, max_episode_steps=100)
env = gym.make_vec("gymnasium_env/Ballz-v0", num_envs=3, width=700, height=950, nblocks=7, render_mode="human", vectorization_mode="async")
#env = gym.make("gymnasium_env/Ballz-v0", width=700, height=950, nblocks=7, render_mode='human')
import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
agent = BallzAgent(env, alpha=2e-4, gamma=0.9, epsilon=0.2)
rewards = agent.train(100, live_plot=False, ppo=False, mc=False)

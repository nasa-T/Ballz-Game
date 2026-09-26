import gymnasium as gym
import numpy as np
from ai_trainer import *

gym.register("gymnasium_env/Ballz-v0", entry_point=BallzEnv, max_episode_steps=100)
env = gym.make("gymnasium_env/Ballz-v0", width=700, height=950, nblocks=7, render_mode='human')
agent = BallzAgent(env)

rewards = agent.train(10, live_plot=True)

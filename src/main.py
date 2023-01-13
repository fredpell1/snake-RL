import gymnasium as gym
import envs
import numpy as np
from gymnasium import spaces
env = envs.SnakeEnv(render_mode="human", size=25)
observation, info = env.reset()


for _ in range(100):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    #print(observation)
    if terminated or truncated:
        observation, info = env.reset()

env.close()

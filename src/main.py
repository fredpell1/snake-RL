import gymnasium as gym
import envs
import numpy as np
from gymnasium import spaces
import pygame
from controller import user_mode, agent_mode


env = envs.SnakeEnv(render_mode="human", size=10)
observation, info = env.reset()


user_mode(env)


exit(0)

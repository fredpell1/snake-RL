import gymnasium as gym
import envs
import numpy as np
from gymnasium import spaces
import pygame
from controller import user_mode, agent_mode
from agents import heuristic_agent

env = envs.SnakeEnv(render_mode="human", size=10)
observation, info = env.reset()

agent = heuristic_agent.HeuristicAgent()

agent_mode(env, 10, 10, agent)



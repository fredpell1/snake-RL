import gymnasium as gym
import envs
import numpy as np
from gymnasium import spaces
import pygame
from controller import user_mode, agent_mode
from agents import heuristic_agent, mc_agent

env = envs.SnakeEnv(render_mode='human', size=10)
observation, info = env.reset()

# agent = heuristic_agent.HeuristicAgent()
agent = mc_agent.MonteCarloNN(0.01, 0.01)
agent_mode(env=env, n_episodes=1000, agent=agent)
agent.save('saved_agent/MC-v0')
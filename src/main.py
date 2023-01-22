import gymnasium as gym
import envs
import numpy as np
from gymnasium import spaces
import pygame
from controller import user_mode, agent_mode
from agents import heuristic_agent, mc_agent
import torch

#load pretrained
env = envs.SnakeEnv(size=10)
observation, info = env.reset()
checkpoint = torch.load("saved_agent/MC-v0")
# agent = mc_agent.MonteCarloNN(0.1,0.1)
# agent.save('saved_agent/MC-v0')
# exit(0)
agent = mc_agent.MonteCarloNN(
    0.1,
    0.5,
    value_function=checkpoint["value_function"],
    optimizer=checkpoint["optimizer_state"],
    loss_function=checkpoint["loss"],
    mode="training",
    learning_rate=0.001
)
#train and save pretrained
agent_mode(env=env, n_episodes=150000, agent=agent, max_step=1000)
agent.save('saved_agent/MC-v0')
print('training')
print(agent.greedy_count, agent.random_count)
print(agent.action_count)

#test trained model
env = envs.SnakeEnv(render_mode="human", size=10)
agent.eval()
agent_mode(env, 10,agent,mode='testing')
print('testing')
print(agent.greedy_count, agent.random_count)
print(agent.action_count)

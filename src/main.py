import gymnasium as gym
import envs
import numpy as np
from gymnasium import spaces
import pygame
from controller import user_mode, agent_mode
from agents import heuristic_agent, mc_agent
import torch

# load pretrained
env = envs.SnakeEnv(size=10)
observation, info = env.reset()
checkpoint = torch.load("saved_agent/MC-v1")
# agent = mc_agent.MonteCarloNN(0.1,0.1)
# agent.save('saved_agent/MC-v1')
# exit(0)
agent = mc_agent.MonteCarloNN(
    0.1,
    0.5,
    value_function=checkpoint["value_function"],
    optimizer=checkpoint["optimizer_state"],
    loss_function=checkpoint["loss"],
    mode="training",
    learning_rate=0.001,
)
# train and save pretrained
for _ in range(3):
    agent_mode(env=env, n_episodes=50000, agent=agent, max_step=1000)
    agent.save("saved_agent/MC-v1")
    print("training")
    print(agent.greedy_count, agent.random_count)
    print(agent.action_count)

# test trained model
env = envs.SnakeEnv(render_mode="human", size=10)
agent.eval()
agent_mode(env, 25, agent, mode="testing", max_step=1000)
print("testing")
print(agent.greedy_count, agent.random_count)
print(agent.action_count)

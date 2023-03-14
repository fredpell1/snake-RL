import envs
import pygame
from agents.base_agent import BaseAgent
import sys
import numpy as np


def user_mode(verbose: bool):
    play = True
    env = envs.SnakeEnv(render_mode="human", size=10)
    env.reset()
    move = env.first_move()

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    play = False

                elif event.key == pygame.K_LEFT:
                    move = 2
                elif event.key == pygame.K_RIGHT:
                    move = 0
                elif event.key == pygame.K_UP:
                    move = 3
                elif event.key == pygame.K_DOWN:
                    move = 1
        observation, reward, target, terminated, info, pixels = env.step(move)
        if verbose:
            print(observation, reward)
            head = observation["agent"]
            body = observation["body"]
            if not terminated:
                grid = np.zeros((10, 10))
                grid[head[1], head[0]] += 2
                grid = grid.flatten()
                print("head coordinate", np.where(grid == 2))
        if terminated:
            env.reset()
            move = env.first_move()
        if target:
            env.eat_apple()
    env.close()


def agent_mode(
    env: envs.SnakeEnv,
    n_episodes: int,
    agent: BaseAgent,
    max_step: int = None,
    mode: str = "training",
    verbose: bool = False,
    fixed_start: bool = False,
):
    max_step = max_step if max_step else sys.maxsize
    rewards = []
    targets = []
    lengths = []
    for _ in range(n_episodes):
        agent.reset()
        if env.render_mode == None:
            observation, info = env.reset(fixed_start=fixed_start)
        else:
            observation, info, pixels = env.reset(fixed_start=fixed_start)
        episode_reward = 0
        episode_targets = 0
        episode_length = 3
        for i in range(max_step):
            action = agent.select_action(observation)
            if env.render_mode == None:
                observation, reward, target, terminated, info = env.step(action)
            else:
                observation, reward, target, terminated, info, pixels = env.step(action)

            episode_reward += reward
            if verbose:
                print(reward)
            if target:
                env.eat_apple()
                episode_targets += 1
                episode_length += 1
            if mode == "training":
                agent.update(reward, observation, action, terminated)
            if terminated:
                break
        rewards.append(episode_reward)
        targets.append(episode_targets)
        lengths.append(episode_length)
    env.close()
    return rewards, agent.losses

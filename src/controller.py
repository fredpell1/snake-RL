import envs
import pygame
from agents.base_agent import BaseAgent
import sys

def user_mode(env: envs.SnakeEnv):
    play = True
    
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
        observation, reward, terminated, truncated, info = env.step(move)
        #print(reward)
        if truncated:
            env.reset()
            move = env.first_move()
        if terminated:
            env.eat_apple()
    env.close()


def agent_mode(env: envs.SnakeEnv, n_episodes: int, max_step: int, agent: BaseAgent):
    max_step = max_step if max_step else sys.maxsize
    rewards = []
    for _ in range(n_episodes):
        agent.reset()
        observation, info = env.reset()
        episode_reward = 0
        for i in range(max_step):
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated:
                observation = env.eat_apple()
            agent.update(reward, observation, action, terminated)
            if truncated:
                break
        rewards.append(episode_reward)
    env.close()
    return rewards


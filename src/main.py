import gymnasium as gym
import envs
import numpy as np
from gymnasium import spaces
import pygame

env = envs.SnakeEnv(render_mode="human", size=10)
observation, info = env.reset()

def playing_loop(env: envs.SnakeEnv):
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
        if truncated:
            env.reset()
            move = env.first_move()
        if terminated:
            env.eat_apple()
    env.close()

playing_loop(env)


exit(0)
for _ in range(100):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation)
    if terminated or truncated:
        observation, info = env.reset()

env.close()

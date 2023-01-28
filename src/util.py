import torch
from controller import agent_mode
import envs

def initialize_new_agent(agent, **kwargs):
    return agent(**kwargs)

def load_agent(agent, filename, **kwargs):
    checkpoint = torch.load(filename)
    value_function=checkpoint["value_function"]
    optimizer=checkpoint["optimizer_state"]
    loss_function=checkpoint["loss"]
    return agent(value_function=value_function, optimizer = optimizer, loss_function = loss_function, **kwargs)

def train_and_save(agent, n_episodes, max_step, filename):
    env = envs.SnakeEnv(size=10)
    _,_ = env.reset()
    agent_mode(env=env, n_episodes=n_episodes, agent=agent, max_step=max_step)
    agent.save(filename)

def test(agent, n_episodes, max_step):
    env = envs.SnakeEnv(render_mode='human', size=10)
    agent.eval()
    agent_mode(env=env, n_episodes=n_episodes, agent=agent, max_step=max_step, mode='testing')
    
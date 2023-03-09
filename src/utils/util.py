import torch
from utils.controller import agent_mode
from utils.wrapper import MultiFrame
import envs
import numpy as np
import yaml
from collections import ChainMap
import re
import os
from agents.td_agent import *
from agents.q_agent import *


def get_agent_type(config_file):
    # hardcode for now but should come up with a more general solution
    mapping = {"TD.*": TDLambdaNN, "DQN.*": DQNAgent}
    for regex, agent in mapping.items():
        if re.search(regex, config_file):
            return agent


def load_config_file(agent, config_file, saved_agent_folder, verbose):
    with open(config_file, "r") as f:
        parameters = yaml.safe_load(f)

    try:
        agent_params = parameters["agent_params"]
        network_params = parameters["network_params"]
        training_params = parameters["training_params"]
        value_function = torch.nn.Sequential()
        agent_params["verbose"] = verbose
        for layer in network_params:
            layer_type = list(layer.keys())[0]
            if hasattr(torch.nn, layer_type):
                value_function.append(getattr(torch.nn, layer_type)(*layer[layer_type]))
            else:
                raise ValueError(f"invalid layer: {layer}")
        training_functions = {}
        for param, value in training_params.items():
            if hasattr(torch.nn, param):
                training_functions["loss_function"] = getattr(torch.nn, param)(*value)
            elif hasattr(torch.optim, param):
                training_functions["optimizer"] = getattr(torch.optim, param)(
                    params=value_function.parameters(), **dict(ChainMap(*value))
                )
        training_functions["value_function"] = value_function
        version = re.search(r".*\/(?P<version>.*)\.yaml", config_file)
        if version:
            version = version.group("version")
            if version in os.listdir(saved_agent_folder):
                checkpoint = torch.load(f"{saved_agent_folder}/{version}")
                training_functions["value_function"].load_state_dict(
                    checkpoint["value_function"]
                )
                training_functions["optimizer"].load_state_dict(
                    checkpoint["optimizer_state"]
                )
                training_functions["loss_function"].load_state_dict(checkpoint["loss"])
                if "epsilon" in checkpoint:
                    agent_params["epsilon"] = checkpoint["epsilon"]
                if "n_steps" in checkpoint:
                    agent_params["n_steps"] = checkpoint["n_steps"]
                if "buffer" in checkpoint:
                    agent_params["buffer"] = checkpoint["buffer"]

        return (
            agent(**agent_params, **training_functions),
            f"{saved_agent_folder}/{version}",
        )

    except KeyError:
        raise ValueError(f"{config_file} is invalid")


def train_and_save(agent, n_episodes, max_step, filename, output_file, verbose=False):
    env = envs.SnakeEnv(size=10)
    if agent.input_type == 'multiframe':
        env = MultiFrame(env, agent.n_frames)
    _ = env.reset()
    rewards,losses = agent_mode(
        env=env, n_episodes=n_episodes, agent=agent, max_step=max_step, verbose=verbose, fixed_start=agent.fixed_start
    )
    agent.save(filename)
    with open(output_file, "ab") as f:
        f.write(b"\n")
        np.savetxt(f, rewards)
    with open(f'losses/dqn-cnn-v2.txt', "ab") as f:
        f.write(b"\n")
        np.savetxt(f,losses)


def test(agent, n_episodes, max_step, verbose):
    env = envs.SnakeEnv(render_mode="human", size=10)
    if agent.input_type == 'multiframe':
        env = MultiFrame(env, agent.n_frames)
    agent.eval()
    agent_mode(
        env=env,
        n_episodes=n_episodes,
        agent=agent,
        max_step=max_step,
        mode="testing",
        verbose=verbose
    )

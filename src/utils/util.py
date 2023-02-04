import torch
from utils.controller import agent_mode
import envs
import numpy as np
import yaml
from collections import ChainMap
import re
import os


def load_config_file(agent, config_file, saved_agent_folder):
    with open(config_file, "r") as f:
        parameters = yaml.safe_load(f)

    try:
        agent_params = parameters["agent_params"]
        network_params = parameters["network_params"]
        training_params = parameters["training_params"]
        value_function = torch.nn.Sequential()
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
        version = re.search(r"\/(?P<version>.*)\.yaml", config_file)
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

        return (
            agent(**agent_params, **training_functions),
            f"{saved_agent_folder}/{version}",
        )

    except KeyError:
        raise ValueError(f"{config_file} is invalid")


def train_and_save(agent, n_episodes, max_step, filename, output_file):
    env = envs.SnakeEnv(size=10)
    _, _ = env.reset()
    rewards = agent_mode(env=env, n_episodes=n_episodes, agent=agent, max_step=max_step)
    with open(output_file, "ab") as f:
        f.write(b"\n")
        np.savetxt(f, rewards)
    agent.save(filename)


def test(agent, n_episodes, max_step):
    env = envs.SnakeEnv(render_mode="human", size=10)
    agent.eval()
    agent_mode(
        env=env, n_episodes=n_episodes, agent=agent, max_step=max_step, mode="testing"
    )
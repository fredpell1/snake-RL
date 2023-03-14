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


def get_agent_type(config_file: str) -> BaseAgent:
    """Determine the agent to load based on config file name

    Args:
        config_file (str): config file name

    Returns:
        BaseAgent: Agent to initialize
    """
    # hardcode for now but should come up with a more general solution
    mapping = {"TD.*": TDLambdaNN, "DQN.*": DQNAgent}
    for regex, agent in mapping.items():
        if re.search(regex, config_file):
            return agent


def load_config_file(
    agent: BaseAgent, config_file: str, saved_agent_folder: str, verbose: bool
) -> tuple:
    """Load config file and build agent at runtime

    Args:
        agent (BaseAgent): Agent Object to initialize
        config_file (str): config file for the agent
        saved_agent_folder (str): Folder where saved models are stored
        verbose (bool): Print more information during training

    Raises:
        ValueError: Config file is not in the right format

    Returns:
        tuple: agent and config file
    """
    with open(config_file, "r") as f:
        parameters = yaml.safe_load(f)

    try:
        agent_params = parameters["agent_params"]
        network_params = parameters["network_params"]
        training_params = parameters["training_params"]
        value_function = build_value_function(network_params)
        training_functions = build_training_functions(training_params, value_function)

        version = get_model_version(config_file)

        if version in os.listdir(saved_agent_folder):
            load_existing_model(
                saved_agent_folder, agent_params, training_functions, version
            )

        # add misc params
        agent_params["verbose"] = verbose

        return (
            agent(**agent_params, **training_functions),
            f"{saved_agent_folder}/{version}",
        )

    except KeyError:
        raise ValueError(f"{config_file} is invalid")


def load_existing_model(
    saved_agent_folder: str, agent_params: dict, training_functions: dict, version: str
):
    """Load parameters of existing model

    Args:
        saved_agent_folder (str): Folder containing saved models
        agent_params (dict): agent's parameters
        training_functions (dict): Loss function, Optimiser, and Value Function
        version (str): Model's version
    """
    checkpoint = torch.load(f"{saved_agent_folder}/{version}")

    # loading core parts
    training_functions["value_function"].load_state_dict(checkpoint["value_function"])
    training_functions["optimizer"].load_state_dict(checkpoint["optimizer_state"])
    training_functions["loss_function"].load_state_dict(checkpoint["loss"])

    # loading misc parameters for specificities of some models
    if "epsilon" in checkpoint:
        agent_params["epsilon"] = checkpoint["epsilon"]
    if "n_steps" in checkpoint:
        agent_params["n_steps"] = checkpoint["n_steps"]
    if "buffer" in checkpoint:
        agent_params["buffer"] = checkpoint["buffer"]


def build_training_functions(
    training_params: dict, value_function: torch.nn.Module
) -> dict:
    """Build the optimizer and the loss function

    Args:
        training_params (dict): Parameters of the loss function and the optimizer
        value_function (torch.nn.Module): Value function from which the optimizer get the parameters to optimize

    Returns:
        (dict): Optimizer, Loss function, and Value Function
    """
    training_functions = {}
    for param, value in training_params.items():
        if hasattr(torch.nn, param):
            training_functions["loss_function"] = getattr(torch.nn, param)(*value)
        elif hasattr(torch.optim, param):
            training_functions["optimizer"] = getattr(torch.optim, param)(
                params=value_function.parameters(), **dict(ChainMap(*value))
            )
    training_functions["value_function"] = value_function
    return training_functions


def build_value_function(network_params: list[dict]) -> torch.nn.Module:
    """Build the Neural Network layer by layer

    Args:
        network_params (list[dict]): Layers and their respective parameters

    Raises:
        ValueError: A layer does not exist in the torch library

    Returns:
        (torch.nn.Module): Neural Network approximating the value function
    """
    value_function = torch.nn.Sequential()
    for layer in network_params:
        layer_type = list(layer.keys())[
            0
        ]  # converting to list because dicts don't support indexing
        if hasattr(torch.nn, layer_type):
            value_function.append(getattr(torch.nn, layer_type)(*layer[layer_type]))
        else:
            raise ValueError(f"invalid layer: {layer}")
    return value_function


def get_model_version(config_file: str) -> str:
    """Extract Model version from config file name

    Args:
        config_file (str): Model's configurations

    Raises:
        ValueError: File not named properly

    Returns:
        str: Model's version
    """
    version = re.search(r".*\/(?P<version>.*)\.yaml", config_file)
    if version:
        version = version.group("version")
    else:
        raise ValueError("Invalid file format")
    return version


def extract_model_from_output_file(output_file: str) -> str:
    """Extract model version from output file to save other metrics in other files

    Args:
        output_file (str): output file

    Raises:
        ValueError: incorrect output file

    Returns:
        str: model version
    """
    model = re.search(r"outputs/(?P<model>.*)\.txt", output_file)
    if model:
        model = model.group("model")
    else:
        raise ValueError("Invalid output file")
    return model


def train_and_save(
    agent: BaseAgent,
    n_episodes: int,
    max_step: int,
    filename: str,
    output_file: str,
    verbose: bool = False,
):
    """Train the agent and saves its parameters

    Args:
        agent (BaseAgent): Agent to train
        n_episodes (int): Number of episodes in training session
        max_step (int): Max step per episode
        filename (str): file where the model's weights are saved
        output_file (str): File where various outputs are saved
        verbose (bool, optional): Print more information. Defaults to False.
    """
    env = envs.SnakeEnv(size=10)
    if agent.input_type == "multiframe":
        env = MultiFrame(env, agent.n_frames)
    _ = env.reset()
    rewards, losses, _, _ = agent_mode(
        env=env,
        n_episodes=n_episodes,
        agent=agent,
        max_step=max_step,
        verbose=verbose,
        fixed_start=agent.fixed_start,
    )
    agent.save(filename)
    with open(output_file, "ab") as f:
        f.write(b"\n")
        np.savetxt(f, rewards)
    model = extract_model_from_output_file(output_file)
    with open(f"losses/{model}.txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f, losses)


def test(agent: BaseAgent, n_episodes: int, max_step: int, verbose: bool = False):
    """Test the agent

    Args:
        agent (BaseAgent): Agent to test
        n_episodes (int): Number of episodes in testing session
        max_step (int): Max steps per episode
        verbose (bool, optional): Prints more information. Defaults to False.
    """
    env = envs.SnakeEnv(render_mode="human", size=10)
    if agent.input_type == "multiframe":
        env = MultiFrame(env, agent.n_frames)
    agent.eval()
    rewards, losses, targets, lengths = agent_mode(
        env=env,
        n_episodes=n_episodes,
        agent=agent,
        max_step=max_step,
        mode="testing",
        verbose=verbose,
    )

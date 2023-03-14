import pytest
import src.utils.util as util
from src.agents.q_agent import DQNAgent
import torch


def test_get_agent_type_match():
    agent = util.get_agent_type("configs/DQN-cnn-v1.txt")
    assert type(agent) == type(DQNAgent)


def test_get_agent_type_no_match():
    agent = util.get_agent_type("configs/MC-v1.txt")
    assert agent is None


def test_get_agent_type_nested_folder():
    agent = util.get_agent_type("configs/dqn/DQN-cnn-v1.txt")
    assert type(agent) == type(DQNAgent)


def test_build_value_function():
    network_parameters = [
        {"Conv2d": [2, 32, 2, 1]},
        {"ReLU": []},
        {"Conv2d": [32, 64, 4, 2]},
        {"ReLU": []},
        {"Flatten": [1, -1]},
        {"Linear": [576, 512]},
        {"ReLU": []},
        {"Linear": [512, 4]},
    ]
    network = util.build_value_function(network_parameters)
    assert isinstance(network, torch.nn.Module) and isinstance(
        network[0], torch.nn.Conv2d
    )


def test_build_training_functions():
    value_function = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU())
    training_params = {"HuberLoss": [], "AdamW": [{"lr": 0.0001}, {"amsgrad": True}]}

    training_functions = util.build_training_functions(training_params, value_function)

    assert (
        training_functions["value_function"] == value_function
        and isinstance(training_functions["loss_function"], torch.nn.HuberLoss)
        and isinstance(training_functions["optimizer"], torch.optim.AdamW)
    )


def test_get_model_version_match():
    config_file = "configs/dqn/DQN-CNN-V1.yaml"
    version = util.get_model_version(config_file)
    assert version == "DQN-CNN-V1"


def test_get_model_version_no_match():
    config_file = "configs/dqn/DQN-CNN-V1.txt"
    with pytest.raises(ValueError):
        version = util.get_model_version(config_file)

def test_extract_model_from_output_file():
    output_file = "outputs/dqn-cnn/dqn-cnn-v1.txt"
    model = util.extract_model_from_output_file(output_file)
    assert model == "dqn-cnn/dqn-cnn-v1"

import pytest
import torch
from src.agents.q_agent import DQNAgent
import numpy as np

@pytest.fixture
def value_function_vector():
    return torch.nn.Sequential(
        torch.nn.Linear(100,50),
        torch.nn.ReLU(),
        torch.nn.Linear(50,4)
    )

@pytest.fixture
def value_function_cnn():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1,32,2,1),
        torch.nn.Conv2d(32,64,4,2),
        torch.nn.Flatten(1,-1),
        torch.nn.Linear(576,512), 
        torch.nn.Linear(512,4)
    )

@pytest.fixture
def value_function_multiframe():
    return torch.nn.Sequential(
        torch.nn.Conv2d(2,32,2,1),
        torch.nn.Conv2d(32,64,4,2),
        torch.nn.Flatten(1,-1),
        torch.nn.Linear(576,512), 
        torch.nn.Linear(512,4)
    )
    

@pytest.fixture
def agent_vector(value_function_vector):
    return DQNAgent(
        buffer_size=10000,
        batch_size=128,
        epsilon=0.30,
        gamma=0.99,
        learning_rate=0.0001,
        tau=0.005,
        epsilon_min=0.05,
        epsilon_decay=2000,
        optimizer=torch.optim.AdamW(value_function_vector.parameters(), 0.0001, amsgrad=True),
        loss_function=torch.nn.HuberLoss(),
        value_function=value_function_vector, 
        input_type='vector'
    )

@pytest.fixture
def agent_grid(value_function_cnn):
    return DQNAgent(
        buffer_size=10000,
        batch_size=128,
        epsilon=0.30,
        gamma=0.99,
        learning_rate=0.0001,
        tau=0.005,
        epsilon_min=0.05,
        epsilon_decay=2000,
        optimizer=torch.optim.AdamW(value_function_cnn.parameters(), 0.0001, amsgrad=True),
        loss_function=torch.nn.HuberLoss(),
        value_function=value_function_cnn, 
        input_type='grid'
    )

@pytest.fixture
def agent_multiframe(value_function_multiframe):
    return DQNAgent(
        buffer_size=10000,
        batch_size=128,
        epsilon=0.30,
        gamma=0.99,
        learning_rate=0.0001,
        tau=0.005,
        epsilon_min=0.05,
        epsilon_decay=2000,
        optimizer=torch.optim.AdamW(value_function_multiframe.parameters(), 0.0001, amsgrad=True),
        loss_function=torch.nn.HuberLoss(),
        value_function=value_function_multiframe, 
        input_type='multiframe',
        n_frames=2
    )

def test_build_grid(agent_vector):
    observation = {
        'agent': np.array([5,5]),
        'body': np.array([[5, 6], [5, 7]]),
        'target': np.array([7,6])
    }
    _,grid = agent_vector._build_grid(observation)
    assert grid[5,5] == 1 and grid[6,5] == 0 and grid[7,5] == 0 and grid[6,7] == 2

def test_build_grid_head_target_overlap(agent_vector):
    observation = {
        'agent': np.array([5,5]),
        'body': np.array([[5, 6], [5, 7]]),
        'target': np.array([5,5])
    }
    _,grid = agent_vector._build_grid(observation)
    assert grid[5,5] == 4

def test_build_grid_head_body_overlap(agent_vector):
    observation = {
        'agent': np.array([5,5]),
        'body': np.array([[5, 5], [5, 6]]),
        'target': np.array([7,6])
    }
    _,grid = agent_vector._build_grid(observation)
    assert grid[5,5] == 0

def test_get_state_from_obs_flatten(agent_vector):
    observation = {
        'agent': np.array([5,5]),
        'body': np.array([[5, 6], [5, 7]]),
        'target': np.array([7,6])
    }
    vector = agent_vector._get_state_from_obs(observation)
    assert vector.shape == (1,100) and vector[0,55] == 1
    
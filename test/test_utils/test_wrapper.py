import pytest
from src.envs.snake import SnakeEnv
from src.utils.wrapper import MultiFrame
import numpy as np


@pytest.fixture
def env():
    env = SnakeEnv(size=10)
    env.reset()
    return env


@pytest.fixture
def multiframe(env):
    return MultiFrame(env, 2)


def test_reset_first_time_same_frame(multiframe):
    frames, _ = multiframe.reset()
    assert np.all(np.all(frames[0][key] == frames[1][key]) for key in frames[0].keys())

import pytest
from src.envs import SnakeEnv

@pytest.fixture
def env():
    env = SnakeEnv(size=10)
    env.reset()
    return env

def test_step_up(env):
    before, info = env.reset()
    after, _,_,_,_ = env.step(3)
    assert before['agent'][0] == after['agent'][0] and \
        before['agent'][1] == after['agent'][1] + 1

def test_step_down(env):
    before, info = env.reset()
    after, _,_,_,_ = env.step(1)
    assert before['agent'][0] == after['agent'][0] and \
        before['agent'][1] == after['agent'][1] - 1

def test_step_left(env):
    before, info = env.reset()
    after, _,_,_,_ = env.step(2)
    assert before['agent'][0] == after['agent'][0] + 1 and \
        before['agent'][1] == after['agent'][1] 

def test_step_right(env):
    before, info = env.reset()
    after, _,_,_,_ = env.step(0)
    assert before['agent'][0] == after['agent'][0] - 1 and \
        before['agent'][1] == after['agent'][1] 
    
def test_eat_apple_body_is_growing(env):
    env.eat_apple()
    assert len(env._body_location) == 3

def test_check_body_hit(env):
    env._head_location = env._body_location[0]
    assert env._check_body_hit() == True


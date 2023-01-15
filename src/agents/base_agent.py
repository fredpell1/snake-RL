import numpy as np

class BaseAgent():
    """Main class for implementing reinforcement learning agents

    It must be extended by a subclass and the select_action method must be 
    overriden. 
    """
    def __init__(self) -> None:
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

    def select_action(self, observation):
        raise NotImplementedError("You should implement this method in a subclass")
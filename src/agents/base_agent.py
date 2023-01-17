import numpy as np
from abc import abstractmethod, ABCMeta

class BaseAgent(metaclass=ABCMeta):
    """Main class for implementing reinforcement learning agents

    It must be extended by a subclass and its methods must be 
    overriden. 
    """
    def __init__(self) -> None:
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

    @abstractmethod
    def select_action(self, observation):
        raise NotImplementedError("You should implement this method in a subclass")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("You should implement this method in a subclass")

    @abstractmethod
    def update(self,reward,observation,action,terminated):
        """This method will be called at every step of an episode after an action is taken,
        it is the agent's responsibility to keep track of the number of iterations, the previously taken actions,
        etc. 

        Raises:
            NotImplementedError: This method should be implemented for all agents
        """
        raise NotImplementedError("You should implement this method in a subclass")


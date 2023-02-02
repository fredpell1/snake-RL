import numpy as np
from abc import abstractmethod, ABCMeta
import torch

class BaseAgent(metaclass=ABCMeta):
    """Main class for implementing reinforcement learning agents

    It must be extended by a subclass and its methods must be
    overriden.
    """

    def __init__(self, value_function, optimizer, loss_function) -> None:
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

        self.value_function : torch.nn.Module = value_function
        self.optimizer : torch.nn.Module = optimizer
        self.loss_function : torch.nn.Module = loss_function

    @abstractmethod
    def select_action(self, observation):
        raise NotImplementedError("You should implement this method in a subclass")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("You should implement this method in a subclass")

    @abstractmethod
    def update(self, reward, observation, action, terminated):
        """This method will be called at every step of an episode after an action is taken,
        it is the agent's responsibility to keep track of the number of iterations, the previously taken actions,
        etc.

        Raises:
            NotImplementedError: This method should be implemented for all agents
        """
        raise NotImplementedError("You should implement this method in a subclass")


    def _get_state_from_obs(self,observation):
        head = observation["agent"]
        target = observation["target"]
        body = observation["body"]
        state_array = np.concatenate((head, target, body[0]), dtype=np.float32)
        return torch.from_numpy(state_array).reshape((-1, 1))


    def select_action(self, observation):
        self.prev_state = self._get_state_from_obs(observation)
        max_value = -100
        max_index = 0
        with torch.no_grad():
            p = np.random.random()
            if p > self.epsilon:
                for action_index, move in self._action_to_direction.items():
                    state_tensor = self.prev_state.clone()
                    state_tensor[:2, :] += torch.from_numpy(move).reshape((-1, 1))
                    value = self.value_function(state_tensor.T)
                    if value.item() > max_value:
                        max_value = value.item()
                        max_index = action_index
            else:
                max_index = np.random.randint(0, 3 + 1)

        return max_index

    def save(self, filename):
        torch.save(
            {
                "value_function": self.value_function.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss": self.loss_function.state_dict(),
            },
            filename,
        )

    def eval(self):
        return None
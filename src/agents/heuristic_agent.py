from agents.base_agent import BaseAgent
import numpy as np

class HeuristicAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        self.prev_move = None

    def select_action(self, observation):
        head = observation["agent"]
        target = observation["target"]
        body = observation["body"]
        move = self.prev_move if self.prev_move else 0
        min_distance = 100
        for key,value in self._action_to_direction.items():
            next_state = head + value
            distance = np.linalg.norm(next_state - target, 1)
            if distance < min_distance:
                move = key
                min_distance = distance
        self.prev_move = move
        return move

    def reset(self):
        self.prev_move = None

    def update(self, reward, observation, action, terminated):
        pass

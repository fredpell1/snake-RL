from agents.base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        self.prev_move = None

    def select_action(self, observation):
        head = observation["agent"]
        target = observation["target"]
        body = observation["body"]
        x_head, y_head = head[0], head[1]
        x_target, y_target = target[0], target[1]
        move = self.prev_move if self.prev_move else 1
        if x_head == x_target:
            if y_head < y_target:
                move = 1
            else:
                move = 3
        else:
            if y_head == y_target:
                if x_head < x_target:
                    move = 0
                else:
                    move = 2

        return move

    def reset(self):
        self.prev_move = None

    def update(self, reward, observation, action, terminated):
        pass

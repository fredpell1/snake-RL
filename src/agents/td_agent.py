from agents.base_agent import BaseAgent
import torch
import numpy as np


class TDLambdaNN(BaseAgent):
    def __init__(
        self,
        epsilon,
        gamma,
        learning_rate,
        lambda_,
        value_function,
        optimizer,
        loss_function,
        mode="training",
        subset_actions=False,
    ) -> None:
        super().__init__(value_function, optimizer, loss_function)
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.mode = mode
        self.subset_actions = subset_actions

    def select_action(self, observation):
        self.prev_state = self._get_state_from_obs(observation)
        with torch.no_grad():
            p = np.random.random()
            if self.epsilon < p:
                actions = (
                    self._subset_actions(observation)
                    if self.subset_actions
                    else self._action_to_direction.keys()
                )
                values = {
                    action: self.value_function(
                        self._get_state_from_obs(self._take_step(observation, action))
                    ).numpy()
                    for action in actions
                }
                return max(values, key=values.get)
            else:
                return self._pick_randomly(observation)

   
    def update(self, reward, observation, action, terminated):
        # we voluntiraly don't zero grad the optimizer to have eligibility traces
        for parameter in self.value_function.parameters():
            if parameter.grad is not None:
                parameter.grad *= self.gamma * self.lambda_

        s_prime = self._get_state_from_obs(observation)
        td_target = torch.tensor(reward) + self.gamma * self.value_function(s_prime)
        loss = self.loss_function(self.value_function(self.prev_state), td_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reset(self):
        return None

    def eval(self):
        self.epsilon /= 100
        self.mode = "testing"

    def _get_state_from_obs(self, observation):
        head = observation["agent"]
        target = observation["target"]
        body = observation["body"]
        vector = torch.full((100, 1), -1.0)
        vector[10 * head[1] - 10 + head[0]] += 2
        vector[10 * target[1] - 10 + target[0]] += 3
        for part in body:
            vector[10 * part[1] - 10 + part[0]] += 1
        return vector.T

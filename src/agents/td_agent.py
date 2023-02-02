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
        mode = 'training'

    ) -> None:
        super().__init__(value_function, optimizer, loss_function)
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.mode = mode


    def update(self, reward, observation, action, terminated):
        #we voluntiraly don't zero grad the optimizer to have eligibility traces
        for parameter in self.value_function.parameters():
            if parameter.grad is not None:
                parameter.grad *= self.gamma * self.lambda_
        
        s_prime = self._get_state_from_obs(observation).T
        td_target = torch.tensor(reward) + self.gamma * self.value_function(s_prime)
        loss = self.loss_function(self.value_function(self.prev_state.T), td_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def reset(self):
        return None


    def eval(self):
        self.epsilon /= 100
        self.mode = "testing"

    def _get_state_from_obs(self, observation):
        return super()._get_state_from_obs(observation)

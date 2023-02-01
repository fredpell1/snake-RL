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
        input_size,
        hidden_size,
        value_function,
        optimizer,
        loss_function,
        mode = 'training'

    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.value_function = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, 1)
        )
        if value_function:
            self.value_function.load_state_dict(value_function)
        self.optimizer = torch.optim.SGD(self.value_function.parameters(), lr=learning_rate)
        if optimizer:
            self.optimizer.load_state_dict(optimizer)
        self.loss_function = torch.nn.MSELoss()
        if loss_function:
            self.loss_function.load_state_dict(loss_function)
        
        self.mode = mode


    def update(self, reward, observation, action, terminated):
        #we voluntiraly don't zero grad the optimizer to consider eligibility traces
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

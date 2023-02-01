from agents.base_agent import BaseAgent
import torch
import numpy as np


class MonteCarloNN(BaseAgent):
    def __init__(
        self,
        epsilon,
        gamma,
        learning_rate=0.01,
        input_size=6,
        hidden_size=50,
        value_function=None,
        optimizer=None,
        loss_function=None,
        mode="training",
    ) -> None:
        super().__init__()
        self.value_function = torch.nn.Sequential(
            # torch.nn.LayerNorm(input_size),
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, 1),
        )
        if value_function:
            self.value_function.load_state_dict(value_function)
        self.epsilon = epsilon if mode == "training" else epsilon #-1
        self.gamma = gamma  # discount factor
        self.state_sequence = []
        self.prev_state = None
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.MSELoss()
        if loss_function:
            self.loss_function.load_state_dict(loss_function)
        self.optimizer = torch.optim.SGD(
            self.value_function.parameters(), lr=self.learning_rate
        )
        if optimizer:
            self.optimizer.load_state_dict(optimizer)
        self.losses = []
        self.mode = mode


    def reset(self):
        self.state_sequence = []

    def update(self, reward, observation, action, terminated):
        state = self.prev_state
        self.state_sequence.append((state, action, reward))
        self.prev_state = observation
        if terminated:  # update the weights
            Gs = self._compute_Gs()
            loss = self._train(Gs)
            self.losses.append(loss)


    def eval(self):
        self.epsilon /= 100
        self.mode = "testing"


    def _compute_Gs(self):
        G = self.state_sequence[-1][-1]
        Gs = torch.zeros(len(self.state_sequence))
        ind = -1
        Gs[ind] += G
        for s in self.state_sequence[-2::-1]:
            ind -= 1
            Gs[ind] += s[-1] + self.gamma * G
            G = Gs[ind]
        return Gs

    def _train(self, Gs):
        self.optimizer.zero_grad()
        y = Gs.reshape((-1, 1))
        Xs = (
            torch.cat([s[0].reshape((-1, 1)) for s in self.state_sequence], dim=-1)
            .type(torch.float32)
            .T
        )
        pred = self.value_function(Xs)
        loss = self.loss_function(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

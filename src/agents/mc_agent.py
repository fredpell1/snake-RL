from agents.base_agent import BaseAgent
import torch
import numpy as np

class MonteCarloNN(BaseAgent):

    def __init__(self, epsilon, gamma, learning_rate=0.01, input_size = 6, hidden_size = 50) -> None:
        super().__init__()
        self.value_function = torch.nn.Sequential(
            torch.nn.Linear(6, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,1)
        )
        self.epsilon = epsilon
        self.gamma = gamma #discount factor
        self.state_sequence = []
        self.prev_state = None
        self.learning_rate = learning_rate
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.value_function.parameters(), lr = self.learning_rate)
        self.losses = []
        

    def select_action(self, observation):
        self.prev_state = self._get_state_from_obs(observation)
        max_value = -100
        max_index = 0
        with torch.no_grad():
            for action_index, move in self._action_to_direction.items():
                state_tensor = self.prev_state.clone()
                state_tensor[:2,:] += torch.from_numpy(move).reshape((-1,1))
                value = self.value_function(state_tensor.T)
                if value.item() > max_value:
                    max_value = value.item()
                    max_index = action_index
        return max_index


    def reset(self):
        self.state_sequence = []

    def update(self, reward, observation, action, terminated):
        state = self.prev_state
        self.state_sequence.append(
            (
                state, action, reward
            )
        )
        self.prev_state = observation
        if terminated: #update the weights
            Gs = self._compute_Gs()
            loss = self._train(Gs)
            self.losses.append(loss)
        
    def save(self, filename):
        torch.save(
            {
                'value_function': self.value_function.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'loss': self.loss_function.state_dict(), 
                'losses': self.losses
            }, filename
        )

    def _get_state_from_obs(self, observation):
        head = observation["agent"]
        target = observation["target"]
        body = observation["body"]
        state_array = np.concatenate((head, target, body[0]), dtype = np.float32)
        return torch.from_numpy(state_array).reshape((-1,1))
        
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
        y = Gs.reshape((-1,1))
        Xs = torch.cat(
            [s[0].reshape((-1,1)) for s in self.state_sequence], dim=-1
        ).type(torch.float32).T
        pred = self.value_function(Xs)
        loss = self.loss_function(pred,y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    
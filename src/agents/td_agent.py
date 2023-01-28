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
        self.value_function = value_function
        self.optimizer = optimizer
        self.value_function = value_function
        self.loss_function = loss_function
        self.mode = mode


    def select_action(self, observation):
        self.prev_state = self._get_state_from_obs(observation)
        max_value = -100
        max_index = 0
        with torch.no_grad():
            p = np.random.random()
            if p > self.epsilon:
                self.greedy_count += 1
                for action_index, move in self._action_to_direction.items():
                    state_tensor = self.prev_state.clone()
                    state_tensor[:2, :] += torch.from_numpy(move).reshape((-1, 1))
                    value = self.value_function(state_tensor.T)
                    if value.item() > max_value:
                        max_value = value.item()
                        max_index = action_index
            else:
                max_index = np.random.randint(0, 3 + 1)
                self.random_count += 1

        self.action_count[max_index] += 1
        return max_index

    def update(self, reward, observation, action, terminated):
        #we voluntiraly don't zero grad the optimizer to consider eligibility traces
        for parameter in self.value_function.parameters():
            parameter *= self.gamma * self.lambda_
        
        s_prime = self._get_state_from_obs(observation)
        td_target = torch.tensor(reward) + self.gamma * self.value_function(s_prime)
        loss = self.loss_function(self.value_function(self.prev_state), td_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def reset(self):
        return None

    def save(self,filename):
        #TODO make this a function in the base agent
        torch.save(
            {
                "value_function": self.value_function.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss": self.loss_function.state_dict(),
                "losses": self.losses,
            },
        filename,
    )

    def eval(self):
        self.epsilon /= 100
        self.mode = "testing"
        self.action_count = [0, 0, 0, 0]
        self.greedy_count = 0
        self.random_count = 0

    def _get_state_from_obs(self,observation):
        #TODO make this a function in the base agent
        head = observation["agent"]
        target = observation["target"]
        body = observation["body"]
        state_array = np.concatenate((head, target, body[0]), dtype=np.float32)
        return torch.from_numpy(state_array).reshape((-1, 1))
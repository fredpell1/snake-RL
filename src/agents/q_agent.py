from agents.base_agent import BaseAgent
import torch
from collections import deque
import numpy as np
import copy
import random
class DQNAgent(BaseAgent):

    def __init__(
        self,
        buffer_size,
        batch_size,
        epsilon,
        gamma,
        learning_rate,
        tau,
        epsilon_min, 
        epsilon_decay,
        optimizer,
        loss_function,
        value_function,
    ) -> None:
        super().__init__(value_function, optimizer, loss_function)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer = deque([], maxlen=buffer_size)
        self.policy_net = value_function
        self.target_net = copy.deepcopy(value_function)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.n_steps = 0


    def _save_transition(self, transition):
        self.buffer.append(transition)

    def _sample(self):
        return random.sample(self.buffer, self.batch_size)

    def select_action(self, observation):
        state = self._get_state_from_obs(observation)
        self.prev_state = state
        p = np.random.random()
        epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            np.exp(-1.0 * self.n_steps / self.epsilon_decay)
        self.n_steps += 1
        if epsilon < p:
            with torch.no_grad():
                return int(self.policy_net(state).max(1)[1].view(1,1).item())
        else:
            return self._pick_randomly(observation)

    def reset(self):
        return None

    def update(self, reward, observation, action, terminated):
        self.buffer.append((
            self.prev_state,
            torch.tensor([[action]]), 
            self._get_state_from_obs(observation), 
            torch.tensor([reward])
            )
        )
        #no update if we haven't collected enough sample
        if len(self.buffer) < self.batch_size:
            return
        transitions = self._sample()
        batch = (*zip(*transitions),)
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        next_state_batch = torch.cat(batch[2])
        reward_batch = torch.cat(batch[3])

        # Q(s,a) for all s,a in (state_batch, action_batch)
        state_action_values = self.policy_net(state_batch).gather(1,action_batch)

        # max Q(s',a') for all s',a' in (next_state_batch, action_batch)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        #compute target
        
        target = self.gamma * next_state_values + reward_batch
        #compute loss
        loss = self.loss_function(state_action_values, target.unsqueeze(1))
        
        #optimize and gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        #update target net weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + \
                target_net_state_dict[key] * (1 - self.tau)
        
        self.target_net.load_state_dict(target_net_state_dict)
    

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


    def eval(self):
        pass #self.epsilon /= 100

    def save(self, filename):
        epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * \
            np.exp(-1.0 * self.n_steps / self.epsilon_decay)
        torch.save(
            {
                "value_function": self.value_function.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss": self.loss_function.state_dict(),
                "epsilon": epsilon
            },
            filename,
        )

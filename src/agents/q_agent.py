from agents.base_agent import BaseAgent
import torch
from collections import deque
import numpy as np
import copy
import random


class DQNAgent(BaseAgent):
    input_types = ["vector", "grid", "multiframe"]

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
        n_steps=0,
        verbose=False,
        buffer=None,
        keep_target=True,
        negative_reward=False,
        orthogonal_moves=True,
        input_type="vector",
        n_frames=None,
        fixed_start=False,
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
        self.buffer = deque([], maxlen=buffer_size) if not buffer else buffer
        self.policy_net = value_function
        self.target_net = copy.deepcopy(value_function)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.n_steps = n_steps
        self.verbose = verbose
        self.keep_target = keep_target
        self.negative_reward = negative_reward
        self.orthogonal_moves = orthogonal_moves
        if not self.orthogonal_moves:
            self.prev_action = None

        assert input_type in self.input_types
        self.input_type = input_type
        self.n_frames = n_frames
        self.fixed_start = fixed_start
        self.losses = []

    def _save_transition(self, transition):
        self.buffer.append(transition)

    def _sample(self):
        return random.sample(self.buffer, self.batch_size)

    def select_action(self, observation):
        state = self._get_state_from_obs(observation)
        self.prev_state = state
        p = np.random.random()
        epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
            -1.0 * self.n_steps / self.epsilon_decay
        )
        self.n_steps += 1
        action = None
        if epsilon < p:  # greedy choice
            with torch.no_grad():
                if self.input_type == "grid":
                    state = state.unsqueeze(0)
                action = int(self.policy_net(state).max(1)[1].view(1, 1).item())
        else:  # random choice
            obs = (
                observation if isinstance(observation, dict) else observation[-1]
            )  # set to last frame if multiframe
            action = self._pick_randomly(obs)

        # prohibit orthogonal moves
        if not self.orthogonal_moves:
            if self._is_valid_action(action, obs):
                self.prev_action = action
                return action
            else:
                if self.prev_action:
                    return self.prev_action
                else:  # first action
                    self.prev_action = self._pick_randomly(obs)
                    return self.prev_action
        else:
            return action

    def _is_valid_action(self, action, observation):
        if action in self._subset_actions(observation):
            return True
        else:
            return False

    def reset(self):
        self.prev_action = None

    def update(self, reward, observation, action, terminated):
        # penalize not reaching the target
        if reward == 0 and self.negative_reward:
            reward = -0.01
        self.buffer.append(
            (
                self.prev_state,
                torch.tensor([[action]]),
                self._get_state_from_obs(observation) if not terminated else None,
                torch.tensor([reward]),
            )
        )
        # no update if we haven't collected enough sample
        if len(self.buffer) < self.batch_size:
            return
        transitions = self._sample()
        batch = (*zip(*transitions),)
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        # keeping non-terminal states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch[2])), dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch[2] if s is not None])

        reward_batch = torch.cat(batch[3])
        if (
            self.input_type == "grid"
        ):  # no need to unsqueeze for multiframe since concat adds a dimension
            state_batch = state_batch.unsqueeze(0).transpose(0, 1)
            non_final_next_states = non_final_next_states.unsqueeze(0).transpose(0, 1)
        # Q(s,a) for all s,a in (state_batch, action_batch)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        # max Q(s',a') for all s',a' in (next_state_batch, action_batch) if not a terminal state
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]
        # compute target

        target = self.gamma * next_state_values + reward_batch
        # compute loss
        loss = self.loss_function(state_action_values, target.unsqueeze(1))
        self.losses.append(loss.item())
        # optimize and gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        # update target net weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)

        self.target_net.load_state_dict(target_net_state_dict)

    def _get_state_from_obs(self, observation):
        head, grid = (
            self._build_grid(observation)
            if self.input_type != "multiframe"
            else self.concat_frames(observation)
        )
        vector = grid.flatten()
        if self.verbose:
            print("head:", head)
            print(torch.where(vector == 1))
        if self.input_type == "vector":
            return vector.unsqueeze(0)
        if self.input_type == "grid" or self.input_type == "multiframe":
            return grid.unsqueeze(0)

    def concat_frames(self, observation):
        grids = torch.stack([self._build_grid(ob)[1] for ob in observation])
        head = observation[-1]["agent"]
        return head, grids

    def _build_grid(self, observation):
        head = observation["agent"]
        target = observation["target"]
        body = observation["body"]
        grid = torch.full((10, 10), -1.0)

        grid[head[1], head[0]] += (
            2 if not any(np.array_equal(head, x) for x in body) else 0
        )
        if np.all(head == target):
            if self.keep_target:
                grid[target[1], target[0]] += 3
        else:
            grid[target[1], target[0]] += 3

        for part in body:
            grid[part[1], part[0]] += 1
        return head, grid

    def eval(self):
        self.epsilon = 0
        self.epsilon_min = 0

    def save(self, filename):
        epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
            -1.0 * self.n_steps / self.epsilon_decay
        )
        torch.save(
            {
                "value_function": self.value_function.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss": self.loss_function.state_dict(),
                "epsilon": epsilon,
                "n_steps": self.n_steps,
                "buffer": self.buffer,
            },
            filename,
        )

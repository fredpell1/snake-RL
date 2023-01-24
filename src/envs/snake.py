# below code inspired from https://github.com/Farama-Foundation/gym-examples/blob/main/gym_examples/envs/grid_world.py
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(1, 2), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(1, 2), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # up
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": self._head_location,
            "body": self._body_location,
            "target": self._target_location,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._head_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's head location uniformly at random
        self._head_location = self.np_random.integers(
            1, self.size - 2, size=2, dtype=int
        )

        # Set the initial body location
        x, y = self._head_location[0], self._head_location[1]
        if y > self.size / 2:
            self._body_location = np.array([[x, y + 1], [x, y + 2]])
        else:
            self._body_location = np.array([[x, y - 1], [x, y - 2]])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        # TODO: add check to make sure target does not overlap with body
        self._target_location = self._head_location
        while np.array_equal(self._target_location, self._head_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # updating the head
        head = self._head_location.copy()
        self._head_location = self._head_location + direction

        # We move the body
        for i, part in enumerate(self._body_location):
            temp = head
            head = part.copy()
            self._body_location[i] = temp

        terminated = self._check_body_hit() or self._check_wall_hit()

        target = np.array_equal(self._head_location, self._target_location)
        if target:
            reward = 1
        elif terminated:
            reward = -1
        else:
            reward = -0.1
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._render_frame()

        return observation, reward, target, terminated, info

    def eat_apple(self):
        # spawn apple randomly
        while np.array_equal(self._target_location, self._head_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        # grow body
        x_head, y_head = self._head_location[0], self._head_location[1]
        x_last, y_last = self._body_location[-1, 0], self._body_location[-1, 1]

        if y_head == y_last:
            if x_head > x_last:
                new_x = x_last + 1
            else:
                new_x = x_last - 1
            new_y = y_last
        else:
            if x_head == x_last:
                if y_head > y_last:
                    new_y = y_last - 1
                else:
                    new_y = y_last + 1
                new_x = x_last
            elif x_head > x_last:
                new_y = y_last
                new_x = x_last - 1
            else:
                new_y = y_last
                new_x = x_last + 1
        self._body_location = np.append(self._body_location, [[new_x, new_y]], 0)
        return self._get_obs()

    def first_move(self):

        if self._head_location[1] > (self.size / 2):
            return 3
        else:
            return 1

    def _check_wall_hit(self) -> bool:
        x, y = self._head_location[0], self._head_location[1]
        return x < 0 or x > (self.size - 1) or y < 0 or y > self.size - 1

    def _check_body_hit(self) -> bool:
        for part in self._body_location:
            if np.array_equal(self._head_location, part):
                return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent's head
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (self._head_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw the body
        for parts in self._body_location:
            pygame.draw.circle(
                canvas,
                (1, 50, 32),
                (parts + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

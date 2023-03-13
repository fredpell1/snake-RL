import envs
from collections import deque
import copy


class MultiFrame:
    def __init__(self, env: envs.SnakeEnv, n_frames: int) -> None:
        self.env = env
        self.render_mode = env.render_mode
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self, **kwargs):
        self.frames.clear()
        output = self.env.reset(**kwargs)
        # append copies of the first state
        for _ in range(self.n_frames):
            self.frames.append(copy.deepcopy(output[0]))
        return self.frames, *output[1:]

    def step(self, action):
        output = self.env.step(action)
        self.frames.append(copy.deepcopy(output[0]))
        return self.frames, *output[1:]

    def eat_apple(self):
        return self.env.eat_apple()

    def close(self):
        return self.env.close()

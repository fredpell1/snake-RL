import envs
from collections import deque

class MultiFrame():

    def __init__(self, env: envs.SnakeEnv, n_frames:int) -> None:
        self.env = env
        self.render_mode = env.render_mode
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self):
        self.frames.clear()
        output = self.env.reset()
        #append copies of the first state
        for _ in range(self.n_frames):
            self.frames.append(output[0])
        return self.frames , *output[1:]


    def step(self, action):
        output = self.env.step(action)
        self.frames.append(output[0])
        return self.frames , *output[1:]

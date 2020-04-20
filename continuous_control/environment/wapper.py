import gym
import numpy as np

class Wrapper(gym.Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.env = env
        self.args = args

    def reset(self):
        state = self.env.reset()
        if self.args.normalize_obs:
            state = (state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low) * 2.0 - 1.0
        return state

    def step(self, action):
        action = 0.5 * (action + 1) * (self.action_space.high - self.action_space.low) + self.action_space.low
        action = np.clip(action, self.action_space.low, self.action_space.high) # avoid numerical error
        next_state, reward, done, info = self.env.step(action)
        if self.args.normalize_obs:
            next_state = (next_state - self.observation_space.low)/(self.observation_space.high-self.observation_space.low) * 2.0 - 1.0
        return next_state, reward, done, info
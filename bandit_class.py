import numpy as np


class FiveArmedBandit:
    def __init__(self):
        self.arms = [
            (-3, 2),
            (4, 7),
            (1, 4),
            (-1, 9),
            (2, 5)
        ]

    def sample_reward(self, action):
        start, end = self.arms[action]
        return np.random.uniform(start, end)
    
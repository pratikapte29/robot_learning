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

        self.iteration = 0

    def sample_reward(self, action):
        self.iteration += 1

        # change distribution after 2000 steps
        if self.iteration >= 2000:
            if action == 1:   # arm 2 (index 1)
                return np.random.uniform(-4, 3)
        

        start, end = self.arms[action]
        return np.random.uniform(start, end)
    
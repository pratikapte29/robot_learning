import numpy as np 

def select_action_uniformly():
    """
    generate an action from 1 to 5
    """
    return np.random.randint(0, 5)


def run_five_armed_bandit(bandit, n_actions: int) -> float:
    """_summary_

    Args:
        bandit : instance of the class FiveArmedBandits
        n_actions (int): nuber of actions to simulate 
    """

    sum_reward = 0

    for i in range(n_actions):
        action = select_action_uniformly()
        reward = bandit.sample_reward(action)
        sum_reward += reward

    return sum_reward / n_actions
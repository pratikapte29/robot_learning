import numpy as np 

def select_action_uniformly():
    """
    generate an action from 1 to 5
    """
    return np.random.randint(0, 5)


def run_five_armed_bandit(bandit, n_actions: int) -> float:
    """Implementing 1.2

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


def run_epsilon_greedy(bandit, n_actions: int, epsilon=0.1):
    """Implementing 1.3

    Args:
        bandit (_type_): instance of the class FiveArmedBandits
        n_actions (int): number of actions to simulate
        epsilon (float, optional): probability of choosing a random action.
          Defaults to 0.1.
    """

    Q = [0, 0, 0, 0, 0]  # initial value estimates of each arm
    N = [0, 0, 0, 0, 0]  # store how many times each arm has been selected

    sum_reward = 0

    for i in range(n_actions):
        if np.random.rand() < epsilon:
            action = np.random.randint(0, 5)
        else:
            action = np.argmax(Q)

        N[action] += 1

        reward = bandit.sample_reward(action)
        sum_reward += reward

        # recursive computation from lecture slide 29
        Q[action] += (reward - Q[action]) / N[action]


        # logging for every 100 actions

        if (i+1) % 100 == 0:
            print(f"\nStep {i+1}")

            print("Action % distribution:")
            for j in range(5):
                print(f"Arm {j+1}: {(N[j] / (i+1)) * 100:.2f}%")

            print(f"Avg reward: {sum_reward / (i+1):.4f}")

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


def run_non_stationary(
        bandit_1, bandit_2, n_actions: int, epsilon=0.1, alpha=0.02, Q=0
        ):
    """Implementing 1.4 and 1.5

    Args:
        bandit_1 (_type_): instance of the class FiveArmedBandits
        bandit_2 (_type_): second instance of the same class
        n_actions (int): number of actions to simulate
        epsilon (float, optional): probability of choosing a random action.
          Defaults to 0.1.
        alpha (float, optional): constant learning rate. 
          Defaults to 0.02.
        Q (int): initialization. Defaults to 0.
    """

    # Sample Average
    Q_1 = [Q] * 5
    N_1 = [0] * 5
    sum_reward_1 = 0

    # Constant Alpha
    Q_2 = [Q] * 5
    N_2 = [0] * 5
    sum_reward_2 = 0

    for i in range(n_actions):

        # Sample average computation 
        if np.random.rand() < epsilon:
            action_1 = np.random.randint(0, 5)
        else:
            action_1 = np.argmax(Q_1)

        reward_1 = bandit_1.sample_reward(action_1)
        sum_reward_1 += reward_1

        N_1[action_1] += 1
        Q_1[action_1] += (reward_1 - Q_1[action_1]) / N_1[action_1]

        # learning rate alpha computtation (non stationary)
        if np.random.rand() < epsilon:
            action_2 = np.random.randint(0, 5)
        else:
            action_2 = np.argmax(Q_2)

        reward_2 = bandit_2.sample_reward(action_2)
        sum_reward_2 += reward_2

        N_2[action_2] += 1
        Q_2[action_2] += alpha * (reward_2 - Q_2[action_2])

        # Logging after every 100 steps
        if (i + 1) % 100 == 0:
            print(f"Step {i+1}")
            print()

            print("[Sample Avg]")
            print("Action % distribution:")
            for j in range(5):
                print(f"Arm {j+1}: {(N_1[j] / (i+1)) * 100:.2f}%")
            print(f"Avg reward: {sum_reward_1 / (i+1):.4f}")
            print()

            print("[Alpha = 0.02]")
            print("Action % distribution:")
            for j in range(5):
                print(f"Arm {j+1}: {(N_2[j] / (i+1)) * 100:.2f}%")
            print(f"Avg reward: {sum_reward_2 / (i+1):.4f}")

from bandit_class import FiveArmedBandit
from experiments import run_five_armed_bandit
from experiments import run_epsilon_greedy


bandit_inst = FiveArmedBandit()
sample_average_reward = run_five_armed_bandit(bandit_inst, 10)

print("The sample average reward is: ", sample_average_reward)

run_epsilon_greedy(bandit=bandit_inst, n_actions=5000, epsilon=1)

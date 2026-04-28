from bandit_class import FiveArmedBandit
from experiments import run_five_armed_bandit
from experiments import run_epsilon_greedy
from experiments import run_non_stationary


bandit_inst = FiveArmedBandit()
bandit_alpha = FiveArmedBandit()
# sample_average_reward = run_five_armed_bandit(bandit_inst, 10)

# print("The sample average reward is: ", sample_average_reward)

# run_epsilon_greedy(bandit=bandit_inst, n_actions=5000, epsilon=0.1)

run_non_stationary(bandit_1=bandit_inst, 
                   bandit_2=bandit_alpha, 
                   n_actions=5000, 
                   epsilon=0.1, 
                   alpha=0.02)



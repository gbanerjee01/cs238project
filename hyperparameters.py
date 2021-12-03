algorithm = "sarsalambda" #qlearning OR sarsa
n_adversaries = 5


false_positive_rate = 0.05
false_negative_rate = 0.15
n_episodes = 100000
test_eps = 1000

exp_dir = "./experiments/" + algorithm + "/"
exp_name = str(n_adversaries) + "advs" + str(false_positive_rate) + "fpr" + str(false_negative_rate) + "fnr" + "_"

exp_file_prefix = exp_dir + exp_name
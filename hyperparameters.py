algorithm = "qlearning" #qlearning OR sarsa
n_adversaries = 0


false_positive_rate = 0.05
false_negative_rate = 0.15
n_episodes = 1000001
test_eps = 100001

exp_dir = "./experiments/" + algorithm + "/"
exp_name = str(n_adversaries) + "advs" + str(false_positive_rate) + "fpr" + str(false_negative_rate) + "fnr" + "_"

exp_file_prefix = exp_dir + exp_name
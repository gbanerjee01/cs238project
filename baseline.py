from CircleOfDeathEnv import CircleOfDeath
from CircleOfDeathAdversarialEnv import CircleOfDeathAdversarial
import numpy as np
import gym
from gym import spaces
import random
from tqdm import tqdm
import utils
import hyperparameters
import argparse

# env = CircleOfDeath()
env = CircleOfDeathAdversarial(hyperparameters.n_adversaries)

test_actions_len = []
test_result = []
test_rewards = []

actions_list = []
for episode in tqdm(range(hyperparameters.test_eps)):
    observation = env.reset()
    # env.render()
    goal_list = observation['exit_goal']
    if goal_list == env.exit_zoneNorth: 
        goal_num = 0
    elif goal_list == env.exit_zoneWest: 
        goal_num = 1
    elif goal_list == env.exit_zoneSouth: 
        goal_num = 2, 
    else: 
        goal_num = 3

    ep_reward = 0
    ep_success = False

    for i in range(50):
        # print(observation)
        action = env.action_space.sample()
        actions_list.append(action)
        observation, reward, done, info = env.step(action)

        ep_reward += reward
        # env.render()
        if done:
            if env.state["cur_loc"] in env.state["exit_goal"]:
                ep_success = True
            # print("Episode finished after {} timesteps".format(t+1))
            break


    test_actions_len.append(len(actions_list))
    test_result.append(ep_success)
    test_rewards.append(ep_reward)
            
# breakpoint()

np.save(hyperparameters.exp_file_prefix + "test_actions_len", test_actions_len)
np.save(hyperparameters.exp_file_prefix + "test_result", test_result)
np.save(hyperparameters.exp_file_prefix + "test_rewards", test_rewards)

print("Testing finished")
#TODO: SAVE LISTS w/ exp name
env.close()
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
q_table = np.zeros([36, 4, env.action_space.n]) #4 end zones

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.5

# For plotting metrics
all_epochs = []
all_penalties = []

episode_rewards = []
episode_iterations = []
episode_result = []

for i in tqdm(range(hyperparameters.n_episodes)):
    new_env = env.reset()
    state = new_env['cur_loc']

    #this code block just helps determine which dimension of the q table to fill; I'm explicitly assigning the known exit location as part of the state so that total state space is actually 6 x 6 x 4 instead of 36. This may or may not be relevant to other algorithms
    goal_list = new_env['exit_goal']

    if goal_list == env.exit_zoneNorth: 
        goal_num = 0
    elif goal_list == env.exit_zoneWest: 
        goal_num = 1
    elif goal_list == env.exit_zoneSouth: 
        goal_num = 2, 
    else: 
        goal_num = 3

    # env.render()
    epochs, penalties, reward, = 0, 0, 0
    done = False

    n_iterations = 0
    episode_reward = 0
    episode_successful = False
    
    #q code referenced from a towardsdatascience article and then adapted
    while not done:
        n_iterations += 1

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state, goal_num]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        # env.render()

        episode_reward += reward
        
        old_value = q_table[state, goal_num, action]
        next_max = np.max(q_table[next_state['cur_loc'], goal_num])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, goal_num, action] = new_value

        # if reward == -10:
        #     penalties += 1

        state = next_state['cur_loc']
        epochs += 1

        if done and env.state["cur_loc"] in env.state["exit_goal"]:
            episode_successful = True

    episode_iterations.append(n_iterations)
    episode_rewards.append(episode_reward)
    episode_result.append(episode_successful)
        
    # if i % 1000 == 0:
    #     # clear_output(wait=True)
    #     print(f"Episode: {i}")

print("Training finished.\n")
# breakpoint()

#TODO Save q_table and three lists
np.save(hyperparameters.exp_file_prefix + "q_table", q_table)
np.save(hyperparameters.exp_file_prefix + "episode_rewards", episode_rewards)
np.save(hyperparameters.exp_file_prefix + "episode_iterations", episode_iterations)
np.save(hyperparameters.exp_file_prefix + "episode_result", episode_result)

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
        action = np.argmax(q_table[observation['cur_loc'], goal_num])
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
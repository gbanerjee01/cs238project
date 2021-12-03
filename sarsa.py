#src https://www.geeksforgeeks.org/sarsa-reinforcement-learning/
from CircleOfDeathEnv import CircleOfDeath
from CircleOfDeathAdversarialEnv import CircleOfDeathAdversarial
import numpy as np
import gym
from gym import spaces
import random
from tqdm import tqdm
import utils
import hyperparameters


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

def choose_action(state, goal_num):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample() # Explore action space
    else:
        action = np.argmax(q_table[state, goal_num]) # Exploit learned values
    return action

episode_rewards = []
episode_iterations = []
episode_result = []

for i in tqdm(range(1, hyperparameters.n_episodes)):
    new_env = env.reset()

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

    state = new_env['cur_loc']
    action = choose_action(state, goal_num)

    epochs, penalties, reward, = 0, 0, 0
    done = False

    n_iterations = 0
    episode_reward = 0
    episode_successful = False
    
    while not done:
        n_iterations += 1
        #get next state
        next_state_allinfo, reward, done, info = env.step(action)
        next_state = next_state_allinfo['cur_loc']

        episode_reward += reward

        #choose next action
        next_action = choose_action(next_state, goal_num)
        
        #update (func to learn q val)
        predict = q_table[state, goal_num, action]
        target = reward + gamma * q_table[next_state, goal_num, next_action]
        q_table[state, goal_num, action] += alpha * (target - predict)

        state = next_state
        action = next_action
        epochs += 1

        if done and env.state["cur_loc"] in env.state["exit_goal"]:
            episode_successful = True

    episode_iterations.append(n_iterations)
    episode_rewards.append(episode_reward)
    episode_result.append(episode_successful)

print("Training finished.\n")

np.save(hyperparameters.exp_file_prefix + "q_table", q_table)
np.save(hyperparameters.exp_file_prefix + "episode_rewards", episode_rewards)
np.save(hyperparameters.exp_file_prefix + "episode_iterations", episode_iterations)
np.save(hyperparameters.exp_file_prefix + "episode_result", episode_result)


test_actions_len = []
test_result = []
test_rewards = []

actions_list = []
for episode in range(1):
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

    for t in range(10):
        action = np.argmax(q_table[observation['cur_loc'], goal_num]) #wait do I need to chagne anything here
        actions_list.append(action)
        observation, reward, done, info = env.step(action)

        ep_reward += reward
        # env.render()
        if done:
            if env.state["cur_loc"] in env.state["exit_goal"]:
                ep_success = True


            test_actions_len.append(len(actions_list))
            test_result.append(ep_success)
            test_rewards.append(ep_reward)

            # print("Episode finished after {} timesteps".format(t+1))
            break
# breakpoint()
env.close()

np.save(hyperparameters.exp_file_prefix + "test_actions_len", test_actions_len)
np.save(hyperparameters.exp_file_prefix + "test_result", test_result)
np.save(hyperparameters.exp_file_prefix + "test_rewards", test_rewards)

#TODO: log performance?
#PERFORMANCE EVAL
print ("Performance: ", reward/hyperparameters.n_episodes)
np.save(hyperparameters.exp_file_prefix + "performance", [reward/hyperparameters.n_episodes])

# print(q_table)
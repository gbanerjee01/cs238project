from CircleOfDeathEnv import CircleOfDeath
from CircleOfDeathAdversarialEnv import CircleOfDeathAdversarial
import numpy as np
import gym
from gym import spaces
import random
from tqdm import tqdm
import utils

# env = CircleOfDeath()
env = CircleOfDeathAdversarial(3)
q_table = np.zeros([36, 4, env.action_space.n]) #4 end zones

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.5

# For plotting metrics
all_epochs = []
all_penalties = []

for i in tqdm(range(1, 101)):
    new_env = env.reset()
    state = new_env['cur_loc']
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
    
    #q code referenced from a towardsdatascience article and then adapted
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state, goal_num]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        # env.render()
        
        old_value = q_table[state, goal_num, action]
        next_max = np.max(q_table[next_state['cur_loc'], goal_num])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, goal_num, action] = new_value

        # if reward == -10:
        #     penalties += 1

        state = next_state['cur_loc']
        epochs += 1
        
    # if i % 1000 == 0:
    #     # clear_output(wait=True)
    #     print(f"Episode: {i}")

print("Training finished.\n")

actions_list = []
for episode in range(1):
    observation = env.reset()
    env.render()
    goal_list = observation['exit_goal']
    if goal_list == env.exit_zoneNorth: 
        goal_num = 0
    elif goal_list == env.exit_zoneWest: 
        goal_num = 1
    elif goal_list == env.exit_zoneSouth: 
        goal_num = 2, 
    else: 
        goal_num = 3

    for t in range(10):
        # print(observation)
        action = np.argmax(q_table[observation['cur_loc'], goal_num])
        actions_list.append(action)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
# breakpoint()
env.close()
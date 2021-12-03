#src the textbook, surprisingly
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
STATE_SPACE = 36
q_table = np.zeros([STATE_SPACE, 4, env.action_space.n]) #4 end zones
n_table = np.zeros([STATE_SPACE, env.action_space.n]) #exponentially decaying visit count for all state action pairs. goals NA

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.5
epsilon_decay = 0.999 #took some number from this github adieu2/sarsa-lambda-frozen-lake/blob/master/sarsa_lambda.py

# For plotting metrics
all_epochs = []
all_penalties = []

def choose_action(state, goal_num):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample() # Explore action space
    else:
        action = np.argmax(q_table[state, goal_num]) # Exploit learned values
    return action

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
    
    while not done:
        #get next state
        next_state_allinfo, reward, done, info = env.step(action)
        n_table[state, action]+=1
        next_state = next_state_allinfo['cur_loc']

        #choose next action
        next_action = choose_action(next_state, goal_num)

        #Sarsa temporal difference update partially applied to every state-action pair according to decaying visit count
        s_thing = reward + gamma * q_table[next_state, goal_num, next_action] - q_table[state, goal_num, action]
        
        #update, I guess (func to learn q val)
        for state in range(0, STATE_SPACE):
            for action in range(0, env.action_space.n):
                q_table[state, goal_num, action] += alpha * s_thing * n_table[state, action]
                n_table[state, action] *= gamma * epsilon_decay
        
        # predict = q_table[state, goal_num, action]
        # target = reward + gamma * q_table[next_state, goal_num, next_action]
        # q_table[state, goal_num, action] += alpha * (target - predict)

        state = next_state
        action = next_action
        epochs += 1

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
        action = np.argmax(q_table[observation['cur_loc'], goal_num]) #wait do I need to chagne anything here
        actions_list.append(action)
        observation, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
# breakpoint()
env.close()

#PERFORMANCE EVAL
print ("Performance: ", reward/hyperparameters.n_episodes)
# print(q_table)
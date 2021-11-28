from CircleOfDeathEnv import CircleOfDeath
import numpy as np
import gym
from gym import spaces
import random
from tqdm import tqdm

env = CircleOfDeath()
q_table = np.zeros([36, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.5

# For plotting metrics
all_epochs = []
all_penalties = []

for i in tqdm(range(1, 2)):
    state = env.reset()['cur_loc']
    env.render()
    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        env.render()
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state['cur_loc']])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # if reward == -10:
        #     penalties += 1

        state = next_state['cur_loc']
        epochs += 1
        
    # if i % 1000 == 0:
    #     # clear_output(wait=True)
    #     print(f"Episode: {i}")

print("Training finished.\n")

for episode in range(1):
    observation = env.reset()
    for t in range(10):
        env.render()
        # print(observation)
        action = np.argmax(observation['cur_loc'])
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
breakpoint()
env.close()
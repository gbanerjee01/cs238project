from CircleOfDeathEnv import CircleOfDeath
import numpy as np
import gym
from gym import spaces

env = CircleOfDeath()

env.reset()
for _ in range(10):
    env.render()
    # print(env.action_space.sample())
    state, reward, done, _ = env.step(env.action_space.sample()) # take a random action
    print(reward)
env.close()
# Define and Train the agent
# model = A2C('CnnPolicy', env).learn(total_timesteps=1000)
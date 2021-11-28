from CircleOfDeathEnv import CircleOfDeath
import numpy as np
import gym
from gym import spaces
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

env = CircleOfDeath()
model = A2C('MlpPolicy', env, verbose=1).learn(total_timesteps=100)

obs = env.reset()

dones = False
while not dones:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# for episode in range(1):
#     observation = env.reset()
#     for t in range(10):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
import gym
import time
import numpy as np

from Model import *

if __name__ == "__main__":
    env = gym.make("SpaceInvaders-v0")
    iterations = 0
    cumulative_reward = 0
    for _ in range(100):
        env.reset()
        done = False
        while not done:
            # env.render()
            # time.sleep(0.01)
            observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
            # print(env.action_space.sample())
            # print(done)
            # print(info)
            # print(step[0].shape) # (210, 160, 3)
            # print(step[0])
            # transformed = transform_image(observation)
            # print(transformed)
            # print(transformed.shape)
            # show_image(transformed)
            cumulative_reward += reward
            iterations += 1
        
    print(iterations/100) # 730.51
    print(cumulative_reward/100) # 159

# import gym
# env = gym.make('SpaceInvaders-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(1000):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if reward > 0:
#             print(t, reward)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break

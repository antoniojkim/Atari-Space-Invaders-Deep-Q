
import sys
import os
import subprocess as sp
from math import inf
import time

import numpy as np
import torch
from math import ceil

from random import shuffle, sample
from SpaceInvadersModel import *
from Trainer import *

import gym


    # 0 : "NOOP",
    # 1 : "FIRE",
    # 2 : "UP",
    # 3 : "RIGHT",
    # 4 : "LEFT",
    # 5 : "DOWN",
    # 6 : "UPRIGHT",
    # 7 : "UPLEFT",
    # 8 : "DOWNRIGHT",
    # 9 : "DOWNLEFT",
    # 10 : "UPFIRE",
    # 11 : "RIGHTFIRE",
    # 12 : "LEFTFIRE",
    # 13 : "DOWNFIRE",
    # 14 : "UPRIGHTFIRE",
    # 15 : "UPLEFTFIRE",
    # 16 : "DOWNRIGHTFIRE",
    # 17 : "DOWNLEFTFIRE",


def train(num_episodes, explore_prob, average=100, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:  ", device)

    env = gym.make("SpaceInvaders-v0")

    model = SpaceInvadersModel(verbose=True).to(device)
    trainer = Trainer(model, device, **kwargs)

    model_instance_directory = "./Attempts/attempt2"
    sp.call(f"mkdir -p {model_instance_directory}", shell=True)

    with open(f"{model_instance_directory}/rewards.csv", "w") as file:
        file.write(f"episode,cumulative_reward\n")
    with open(f"{model_instance_directory}/mean_rewards.csv", "w") as file:
        file.write(f"episode,cumulative_reward\n")

    h, w = transform_image_shape

    cumulative_rewards = []
    max_mean = 0
    for episode in range(num_episodes):

        cumulative_reward = 0
        short_term = Memory(None, buckets=True)
        prev_state = None
        current_state = []

        action = env.action_space.sample()
        current_reward = 0

        num_lives_left = 3
        skip_frames = 45
        observation = env.reset()
        done = False


        while not done:
            # print(f"Game: {game}      Iteration: {iteration}")
            # env.render()

            if len(current_state) >= 4:
                if prev_state is not None:
                    short_term.remember([prev_state, action, current_reward, False, current_state])

                prev_state = current_state
                current_state = []
                current_reward = 0

                # start = current_time_milli()
                # loss = trainer.experience_replay()
                # end = current_time_milli()
                # if loss is not None:
                #     print("\n", (end-start)/1000.0)
                #     exit(1)

                if np.random.random() < explore_prob*(num_episodes-episode)/num_episodes:
                    action = env.action_space.sample()
                else:
                    rewards = model.predict(prev_state, device)
                    action = argmax(rewards)

            if skip_frames > 0:
                skip_frames -= 1
            else:
                transformed = transform_image(observation)
                current_state.append(transformed)

            observation, reward, done, info = env.step(action)
            current_reward += reward
            cumulative_reward += reward

            if int(info['ale.lives']) < num_lives_left:
                num_lives_left -= 1

                if num_lives_left < 1 and len(current_state) >= 4:
                    short_term.remember([prev_state, action, current_reward, True, current_state])

                skip_frames = 23
                current_state = []
                current_reward = 0
                action = env.action_space.sample()

            if done:

                discounted_reward = 0
                for i in range(len(short_term)-1, -1, -1):
                    discounted_reward = 0.95 * discounted_reward + short_term[i][2]

                    short_term[i][2] = discounted_reward

                print(f"\rEpisode: {episode}".ljust(20), end="")

                with open(f"{model_instance_directory}/rewards.csv", "a") as file:
                    file.write(f"{episode},{cumulative_reward}\n")

                cumulative_rewards.append(cumulative_reward)
                remember = True
                mean = np.mean(cumulative_rewards[-average:])
                print(f"  Mean:  {round(mean, 4)}".ljust(20), end="")
                with open(f"{model_instance_directory}/mean_rewards.csv", "a") as file:
                    file.write(f"{episode},{mean}\n")

                if cumulative_reward < (mean-0.5*np.std(cumulative_rewards[-average:])):
                    remember = False

                if mean > max_mean:
                    model.save_weights(f"{model_instance_directory}/max_space_invaders_model")
                    max_mean = mean

                if remember:
                    trainer.memory.remember(short_term.memory)

                loss = trainer.experience_replay()

    
    model.save_weights(f"{model_instance_directory}/trained_space_invaders_model")




if __name__ == "__main__":
    train(**{
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "explore_prob": 0.15,
        "discount": 0.95,
        "max_memory_size": 150,
        "batch_size": 50,
        "mini_batch_size": 32,
        "num_episodes": 10000
    })
            

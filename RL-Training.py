
import sys
import os
import subprocess as sp
from math import inf
import time

import numpy as np
import torch
from math import ceil

from random import shuffle, sample
from Model import *

import gym


max_memory = 15000

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


def train():

    max_cumulative_reward = 0
    explore_prob = 0.05
    gamma = 0.9
    game = 1
    iteration = 1

    env = gym.make("SpaceInvaders-v0")

    long_term_memory = []

    def save_short_term(cumulative_reward, short_term_memory):
        if cumulative_reward > max_cumulative_reward/2:
            model.save_weights(f"{model_instance_directory}/model_average")
            print("✓")

            long_term_memory.extend(short_term_memory)
            if len(long_term_memory) > max_memory:
                del long_term_memory[0:len(long_term_memory)-max_memory]


    model_instance_directory = "./Attempts/attempt10"
    sp.call(f"mkdir -p {model_instance_directory}", shell=True)
    sp.call(f"touch {model_instance_directory}/log.csv", shell=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:  ", device)

    with open(f"{model_instance_directory}/log.csv", "w") as file:
        file.write(f"game,iteration,loss,reward,cumulative_reward,done,lives_left\n")

    h, w = transform_image_shape

    # if os.path.isfile(f"{model_instance_directory}/model_max_cumulative_reward"):
    #     model = Model(f"{model_instance_directory}/model_max_cumulative_reward", verbose=True).to(device)
    # else:
    model = Model(verbose=True).to(device)
    
    criterion = torch.nn.MSELoss()

    # if iteration < 20000:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # elif iteration < 40000:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # else:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    while True:

        cumulative_reward = 0
        short_term_memory = []

        num_lives_left = 3
        skip_frames = 45
        observation = env.reset()
        done = False

        while not done:
            # print(f"Game: {game}      Iteration: {iteration}")
            # env.render()
            transformed = transform_image(observation)
            if np.random.random() < explore_prob:
                action = env.action_space.sample()
            else:
                rewards = model.predict(transformed, device)
                action = np.random.choice(np.flatnonzero(rewards == rewards.max()))

            observation, reward, done, info = env.step(action)

            if int(info['ale.lives']) < num_lives_left:
                num_lives_left -= 1
                skip_frames = 23
                print(f"\rReward: {int(cumulative_reward)}".ljust(17), f"Lives: {num_lives_left}".ljust(10), end="")


            # if reward > 0 or np.random.random() > 0.25:
            if skip_frames > 0:
                skip_frames -= 1
            else:
                # idea: store cumulative reward and how far into game it achieved it
                short_term_memory.append((transformed, action, reward, done, transform_image(observation)))

            cumulative_reward += reward

            if reward > 0:
                print(f"\rReward: {int(cumulative_reward)}".ljust(17), f"Lives: {num_lives_left}".ljust(10), end="")

            batches = []
            num_transitions = len(long_term_memory)
            for _ in range(np.random.randint(1, 3)):
                if num_transitions > 64:
                    transitions = sample(long_term_memory, 32)
                    shuffle(transitions)

                    predictions = model.predict([t[4] for t in transitions], device)

                    batch_input = []
                    batch_label = []
                    for t, p in zip(transitions, predictions):
                        state, action, reward, t_done, next_state = t
                        batch_input.append(state)
                        label = np.empty(model_output_size)
                        if t_done:
                            label.fill(reward)
                        else:
                            label.fill(reward+gamma*p.max())

                        batch_label.append(label)

                    batches.append((np.reshape(np.array(batch_input), (-1, 1, h, w)), np.array(batch_label)))

                    num_transitions -= 32

            running_loss = 0
            for batch_input, batch_label in batches:
                input_tensor = torch.from_numpy(batch_input).double().to(device)
                label_tensor = torch.from_numpy(batch_label).double().to(device)
        
                outputs = model.forward(input_tensor)
                optimizer.zero_grad()
                loss = criterion(outputs, label_tensor)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()


            if len(batches) > 0:
                loss = running_loss / len(batches)
            else:
                loss = ""

            with open(f"{model_instance_directory}/log.csv", "a") as file:
                file.write(f"{game},{iteration},{loss},{reward},{cumulative_reward},{num_lives_left}\n")



                # print(f"Loss:               {loss}")
                # print(f"Reward:             {reward}")
                # print(f"Cumulative Reward:  {cummulative_reward}")
                # print(f"Lives Left:         {info['ale.lives']}")
                # print(f"Num Transitions:    {len(transitions)}")
                # print(f"Batch Size:         {len(batches)}")
                
                # if loss > 10000 and iteration > 100:
                #     return

            iteration += 1
        
        game += 1
        
        if cumulative_reward > max_cumulative_reward:
            model.save_weights(f"{model_instance_directory}/model_max_cumulative_reward")
            max_cumulative_reward = cumulative_reward
            
        print("  Max Reward: ", str(int(max_cumulative_reward)).ljust(7), end="")

        save_short_term(cumulative_reward, short_term_memory)




if __name__ == "__main__":

    while True:
        try:
            train()
        except ValueError as e:
            print(e)
            time.sleep(5)
            

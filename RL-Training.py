
    
import subprocess as sp
from math import inf
import time

import numpy as np
import torch
from math import ceil

from random import shuffle, sample
from Model import *

import gym


max_number_of_transitions = 2500

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
    env = gym.make("SpaceInvaders-v0")
    env.reset()

    transitions = []

    def add_to_transitions(state, action, reward, done, next_state):
        transitions.append((state, action, reward, done, next_state))
        if len(transitions) > max_number_of_transitions:
            del transitions[0]


    model_instance_directory = "./Attempts/attempt6"
    sp.call(f"mkdir -p {model_instance_directory}", shell=True)
    sp.call(f"touch {model_instance_directory}/log.csv", shell=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:  ", device)

    with open(f"{model_instance_directory}/log.csv", "w") as file:
        file.write(f"game,iteration,loss,reward,cumulative_reward,lives_left\n")

    model = Model("./Attempts/attempt5/model_max_cumulative_reward").to(device)

    min_loss = inf
    max_cumulative_reward = -inf
    explore_prob = 0.02
    gamma = 0.9
    game = 1
    iteration = 1

    c, h, w = transform_image_shape
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    while True:
        cumulative_reward = 0
        observation = env.reset()
        done = False
        while not done:
            # print(f"Game: {game}      Iteration: {iteration}")
            # env.render()
            transformed = transform_image(observation)
            rewards = model.predict(transformed, device)
            if np.random.random() < explore_prob:
                action = env.action_space.sample()
            else:
                action = np.random.choice(np.flatnonzero(rewards == rewards.max()))

            observation, reward, done, info = env.step(action)

            # if reward > 0 or np.random.random() > 0.25:
                
            add_to_transitions(transformed, action, reward, done, transform_image(observation))
            cumulative_reward += reward

            batches = []
            num_transitions = len(transitions)
            for i in range(np.random.randint(1, 4)):
                if num_transitions >= 32:
                    transition_sample = sample(transitions, 32)
                    batch_input = []
                    batch_label = []
                    for state, action, reward, done, next_state in transition_sample:
                        batch_input.append(state)
                        label = np.empty(model_output_size)
                        if done:
                            label.fill(reward)
                        else:
                            label.fill(reward+gamma*model.predict(next_state, device).max())

                        batch_label.append(label)

                    batches.append((np.reshape(np.array(batch_input), (-1, c, h, w)), np.array(batch_label)))

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

                # print("    Loss: ", loss)
                with open(f"{model_instance_directory}/log.csv", "a") as file:
                    file.write(f"{game},{iteration},{loss},{reward},{cumulative_reward},{info['ale.lives']}\n")

                # if loss < min_loss:
                #     model.save_weights(f"{model_instance_directory}/model_min_loss")
                #     min_loss = loss
                if cumulative_reward > max_cumulative_reward:
                    model.save_weights(f"{model_instance_directory}/model_max_cumulative_reward")
                    max_cumulative_reward = cumulative_reward
                    print("max_cumulative_reward:  ", max_cumulative_reward)

                model.save_weights(f"{model_instance_directory}/model_most_recent")

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




if __name__ == "__main__":

    # while True:
    #     try:
    train()
            # time.sleep(5)
        # except:
        #     pass
            

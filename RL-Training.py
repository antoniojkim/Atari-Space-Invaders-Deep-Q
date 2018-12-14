
    
import subprocess as sp
from math import inf
import time

import numpy as np
import torch
from math import ceil

from random import shuffle, sample
from Model import *

import gym


max_number_of_transitions = 3500
max_number_of_transitions_w_rewards = 1500

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

    transitions = []
    transitions_w_rewards = []

    def add_to_transitions(state, action, reward, done, next_state):
        if reward > 0:
            transitions_w_rewards.append((state, action, reward, done, next_state))
            if len(transitions_w_rewards) > max_number_of_transitions_w_rewards:
                del transitions_w_rewards[0]
        else:
            transitions.append((state, action, reward, done, next_state))
            if len(transitions) > max_number_of_transitions:
                del transitions[0]


    model_instance_directory = "./Attempts/attempt7"
    sp.call(f"mkdir -p {model_instance_directory}", shell=True)
    sp.call(f"touch {model_instance_directory}/log.csv", shell=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:  ", device)

    with open(f"{model_instance_directory}/log.csv", "w") as file:
        file.write(f"game,iteration,loss,reward,cumulative_reward,done,lives_left\n")

    max_cumulative_reward = 0
    explore_prob = 0.025
    gamma = 0.9
    game = 1
    iteration = 1

    c, h, w = transform_image_shape
    
    criterion = torch.nn.MSELoss()

    try:
        model = Model(f"{model_instance_directory}/model_max_cumulative_reward").to(device)
    except:
        model = Model().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    while True:

        cumulative_reward = 0
        num_lives_left = 3
        skip_frames = 0
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

            if int(info['ale.lives']) < num_lives_left:
                num_lives_left -= 1
                skip_frames = 23

            if done and num_lives_left > 0:
                raise ValueError("Inconsistent step error")

            # if reward > 0 or np.random.random() > 0.25:
            if skip_frames > 0:
                skip_frames -= 1
            else:
                add_to_transitions(transformed, action, reward, done, transform_image(observation))

            cumulative_reward += reward

            batches = []
            num_transitions = len(transitions)+len(transitions_w_rewards)
            for _ in range(np.random.randint(1, 2)):
                if num_transitions >= 32:
                    transition_sample = sample(transitions, 32-min(len(transitions_w_rewards), 8))+\
                                        sample(transitions_w_rewards, min(len(transitions_w_rewards), 8))
                    shuffle(transition_sample)

                    batch_input = []
                    batch_label = []
                    for state, action, reward, t_done, next_state in transition_sample:
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

                with open(f"{model_instance_directory}/log.csv", "a") as file:
                    file.write(f"{game},{iteration},{loss},{reward},{cumulative_reward},{done},{num_lives_left}\n")

                if cumulative_reward > max_cumulative_reward:
                    model.save_weights(f"{model_instance_directory}/model_max_cumulative_reward")
                    max_cumulative_reward = cumulative_reward

            else:
                with open(f"{model_instance_directory}/log.csv", "a") as file:
                    file.write(f"{game},{iteration},,{reward},{cumulative_reward},{done},{num_lives_left}\n")


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
        print("Reward: ", cumulative_reward, "   Max Reward: ", max_cumulative_reward, "  ", done)




if __name__ == "__main__":

    # while True:
        # try:
    train()
        # except ValueError as e:
        #     print(e)
        #     time.sleep(5)
            

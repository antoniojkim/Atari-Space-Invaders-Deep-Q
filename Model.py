
import time

import torch
import torch.nn.functional as F
import numpy as np
from numba import jit
import random

import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def transform_image(image):

    # image = image[0:-15, 0:-10, :]
    image = image[23:-15, 10:-10, :]
    # image = np.rollaxis(image, 2, 0)

    return rgb2gray(image) # image/255

transform_image_shape = transform_image(np.zeros((210, 160, 3))).shape


def show_image(image):
    # imgplot = plt.imshow(np.rollaxis(image, 0, 3))
    plt.imshow(image, cmap = plt.get_cmap('gray'))
    plt.show()


def conv_out_shape(input, kernel, stride):
    c = 1 # input[0]
    h = (input[0]-(kernel-1)-1)/stride+1
    w = (input[1]-(kernel-1)-1)/stride+1
    if h%1 != 0:
        print(input, "->", (c, h, w))
        raise Exception("Height out is not an integer")
    if w%1 != 0:
        print(input, "->", (c, h, w))
        raise Exception("Width out is not an integer")
    return int(h), int(w)


model_output_size = 6

class Model(torch.nn.Module):

    def __init__(self, state_dict_path=None, verbose=False):

        super(Model, self).__init__()

        input_shape = transform_image_shape
        if verbose: print("input:      ", input_shape)
        conv1_kernel = 8
        conv1_stride = 4
        conv1_out_channels = 16

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=conv1_out_channels, kernel_size=conv1_kernel, stride=conv1_stride).double()

        conv1_output_shape = conv_out_shape(input_shape, conv1_kernel, conv1_stride)
        if verbose: print("conv1 out:  ", conv1_output_shape)
        conv2_kernel = 4
        conv2_stride = 2
        conv2_out_channels = 32

        self.conv2 = torch.nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=conv2_kernel, stride=conv2_stride).double()

        h, w = conv_out_shape(conv1_output_shape, conv2_kernel, conv2_stride)
        if verbose: print("conv2 out:  ", (h, w))

        self.view_size = h*w*conv2_out_channels
        if verbose: print("view size:  ", self.view_size)
        
        self.fc1 = torch.nn.Linear(self.view_size, 256).double()
        self.fc2 = torch.nn.Linear(256, model_output_size).double()

        if state_dict_path is not None:
            self.load_weights(state_dict_path)

    
    def load_weights(self, state_dict_path: str):
        self.load_state_dict(torch.load(state_dict_path))
        self.eval()

    def save_weights(self, state_dict_path: str):
        torch.save(self.state_dict(), state_dict_path)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.view_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    def predict(self, x, device=None):
        h, w = transform_image_shape
        if device is None:
            return self.forward(torch.tensor(torch.from_numpy(np.reshape(x, (-1, 1, h, w))), dtype=torch.double)).detach().numpy()

        return self.forward(torch.tensor(torch.from_numpy(np.reshape(x, (-1, 1, h, w))), dtype=torch.double).to(device)).cpu().detach().numpy()


current_time_milli = lambda: int(round(time.time() * 1000))

if __name__ == "__main__":

    import gym

    env = gym.make("SpaceInvaders-v0")
    transitions = []
    skip_frames = 45
    observation = env.reset()
    for i in range(45+32):
        if skip_frames > 0:
            skip_frames -= 1
        else:
            transitions.append(transform_image(observation))
        observation, reward, done, info = env.step(env.action_space.sample())

    print(len(transitions))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:  ", device)

    model = Model(verbose=True).to(device)

    # start = current_time_milli()

    # for _ in range(100):
    #     for transition in transitions:
    #         model.predict(transition, device)

    # end = current_time_milli()

    # print((end-start)/1000.0)

    # start = current_time_milli()

    # for _ in range(100):
    #     model.predict(transitions, device)

    # end = current_time_milli()

    # print((end-start)/1000.0)

    print(model.predict(transitions[0], device))

    # print(transform_image_shape)
    # print(transform_image(observation))
    # plt.imshow(transform_image(observation), cmap = plt.get_cmap('gray'))
    # plt.show()


    # start = current_time_milli()

    # # 3.876
    # # 4.101

    # for _ in range(10000):
    #     transform_image(np.random.randint(256, size=(210, 160, 3)))

    # end = current_time_milli()

    


import time

import torch
import torch.nn.functional as F
import numpy as np
from numba import jit, vectorize
import random

import matplotlib.pyplot as plt

import gym
from Trainer import *

# @jit(cache=True)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], np.array([0.299, 0.587, 0.114]))

# @jit(cache=True)
def transform_image(image):

    # image = image[0:-15, 0:-10, :]
    image = image[27:-19, 30:-30, :]
    # image = np.rollaxis(image, 2, 0)

    image = rgb2gray(image)
    image[image > 0] = 1

    return image.astype(np.bool_) # image/255

transform_image_shape = transform_image(np.zeros((210, 160, 3))).shape


def show_image(image):
    # imgplot = plt.imshow(np.rollaxis(image, 0, 3))
    plt.imshow(image*255, cmap = plt.get_cmap('gray'))
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


model_input_size = 4
model_output_size = 6

class SpaceInvadersModel(BaseModel):

    def __init__(self, state_dict_path=None, verbose=False):

        super(SpaceInvadersModel, self).__init__()

        input_shape = transform_image_shape
        if verbose: print("input:      ", input_shape)
        conv1_kernel = 8
        conv1_stride = 4
        conv1_out_channels = 16

        self.conv1 = torch.nn.Conv2d(in_channels=model_input_size, out_channels=conv1_out_channels, kernel_size=conv1_kernel, stride=conv1_stride).double()

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

    def transform(self, x):
        return np.reshape(x, (-1, model_input_size, transform_image_shape[0], transform_image_shape[1])).astype(np.double)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.view_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

if __name__ == "__main__":

    # 13.248, 13.688
    
    # start = current_time_milli()
    env = gym.make("SpaceInvaders-v0")
    observation = env.reset()
    for i in range(200):
        observation, reward, done, info = env.step(4)

    model = SpaceInvadersModel(verbose=True)

    transformed = transform_image(observation)
    # for i in range(100):
    #     compress(transformed)
    print(transformed.shape)
    print(transformed.size)
    print(transformed.size * transformed.itemsize)

    show_image(transformed)

    # end = current_time_milli()

    # print((end-start)/1000.0)

import torch
import torch.nn.functional as F
import numpy as np
import random

import matplotlib.pyplot as plt


def transform_image(image):

    image = image[23:-15, 10:-10, :]
    image = np.rollaxis(image, 2, 0)

    return image/255

transform_image_shape = transform_image(np.zeros((210, 160, 3))).shape


def show_image(image):
    imgplot = plt.imshow(np.rollaxis(image, 0, 3))
    plt.show()


def conv_out_shape(input, kernel, stride):
    c = input[0]
    h = (input[1]-(kernel-1)-1)/stride+1
    w = (input[2]-(kernel-1)-1)/stride+1
    if h%1 != 0:
        print(input, "->", (c, h, w))
        raise Exception("Height out is not an integer")
    if w%1 != 0:
        print(input, "->", (c, h, w))
        raise Exception("Width out is not an integer")
    return int(c), int(h), int(w)


model_output_size = 6

class Model(torch.nn.Module):

    def __init__(self, state_dict_path=None, verbose=False):

        super(Model, self).__init__()

        input_shape = transform_image_shape
        if verbose: print("input:      ", input_shape)
        conv1_kernel = 8
        conv1_stride = 4

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=conv1_kernel, stride=conv1_stride).double()

        conv1_output_shape = conv_out_shape(input_shape, conv1_kernel, conv1_stride)
        if verbose: print("conv1 out:  ", conv1_output_shape)
        conv2_kernel = 4
        conv2_stride = 2

        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=conv2_kernel, stride=conv2_stride).double()

        c, h, w = conv_out_shape(conv1_output_shape, conv2_kernel, conv2_stride)
        if verbose: print("conv2 out:  ", (c, h, w))

        self.view_size = h*w*32
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
        c, h, w = transform_image_shape
        if device is None:
            return self.forward(torch.tensor(torch.from_numpy(np.reshape(x, (-1, c, h, w))), dtype=torch.double)).detach().numpy()

        return self.forward(torch.tensor(torch.from_numpy(np.reshape(x, (-1, c, h, w))), dtype=torch.double).to(device)).cpu().detach().numpy()



if __name__ == "__main__":

    model = Model(verbose=True)

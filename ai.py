# AI for Doom

# Library Import for NN

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Package import for OpenAI and Doom

import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Import other python files
import experience_replay, image_preprocessing


# <Build the AI>

class CNN(nn.Module): # Convolutional Neural Network
    
    def __init__(self, numberOfActions):
        super(CNN, self).__init__()
        
        # 3 convolution layers
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) # in_channels: input image type (b&w = 1, colour = 3), out_channel: number of convoluted images (features you want detected, 32 is common), Kernel_size: detection filter size (5x5) 
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) # in_channel needs to match the previous output size (32 for convolution1), kernel needs to get smaller because previous one is 5x5
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 2) # same here
        
        # Hidden Layer
        self.fc1 = nn.Linear(in_features= numberOfNeurons, out_features=50)
        
        # Output Layer
        self.fc2 = nn.Linear(in_features=50, out_features= numberOfActions) # each node correspondes to 1 q-value which also corresponds to 1 action, so size of output node must match "numberOfActions"
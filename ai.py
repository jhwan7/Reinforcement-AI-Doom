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
        
        # 3 convolution layers, create feature maps based on the initial image
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) # in_channels: input image type (b&w = 1, colour = 3), out_channel: number of convoluted images (features you want detected, 32 is common), Kernel_size: detection filter size (5x5) 
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) # in_channel needs to match the previous output size (32 for convolution1), kernel needs to get smaller because previous one is 5x5
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 2) # same here
        
        # Hidden Layer
        self.fc1 = nn.Linear(in_features= self.countTotalNeurons((1, 80, 80)), out_features=50)
        
        # Output Layer
        self.fc2 = nn.Linear(in_features=50, out_features= numberOfActions) # each node correspondes to 1 q-value which also corresponds to 1 action, so size of output node must match "numberOfActions"
        
    def countTotalNeurons(self, imageDimensions):
        x = Variable(torch.rand(1, *imageDimensions)) # Create a torch variable that holds the size of the initial image passed from Doom (80 x 80)
        
        # pool the feature maps generated in each convolutional layer
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) # pass the variable to calculate the resulting size after it is pooled in c1. then activate it using relu()
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        
        # flatten
        return x.data.view(1, -1).size(1)
    
    def forward(self, input):
        # using the input image, process it throught 3 convolutional layers
        input = F.relu(F.max_pool2d(self.convolution1(input), 3, 2))
        input = F.relu(F.max_pool2d(self.convolution2(input), 3, 2))
        input = F.relu(F.max_pool2d(self.convolution3(input), 3, 2))
        
        # flatten processed data into 1 dimensional array
        flatten = input.view(input.size(0), -1)
        
        # propagate flatten data throught the hidden layer
        hiddenLayer = F.relu(self.fc1(flatten))
        
        # output layer
        return self.fc2(hiddenLayer)
    
class SoftmaxBody(nn.Module):
    
    def __init__(self, temperature):
        super(SoftmaxBody, self).__init__()
        self.temperature = temperature
        
    def determineAction(self, outputs):
        # using each q-value from the output layer, create a probability distribution
        probs = F.softmax(outputs * self.temperature)
        
        # select an action from the probability
        action = probs.multinomial()
        return action
        


class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
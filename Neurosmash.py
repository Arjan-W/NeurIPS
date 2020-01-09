import numpy as np
import socket
from PIL import Image
import torch
import random

from collections import deque

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent:
    def __init__(self):
        self.model = Network().double()
        self.memory = deque(maxlen=500)
        self.iteration = 0
        
        self.epsilon = 1
        self.eps_decay = 0.05
        self.eps_min = 0.09
        
        self.gamma = 0.95
        pass
    
    def state2image(self, state):
        return Image.fromarray(np.array(state, "uint8").reshape(150, 150, 3))
    
    def step(self, end, reward, state):
        #print("step")
        action = 0
        if ( random.uniform(0, 1) <= self.epsilon):
            #explore
            action = random.randint(1, 2)
        else:
            #exploit
            
            #process the state of the game
            #print(np.expand_dims(np.array(self.state2image(state).convert('L'), dtype='d'), axis=0).shape)
            data = np.expand_dims(np.swapaxes(np.array(self.state2image(state), dtype='d'), 0, 2), axis=0)
            #print(data.shape)
            #predict with neural network
            x = torch.tensor(data, dtype=torch.double)
            #print(x)
            predict = self.model.forward(x).data
            
            #extract action
            if(predict[0]>predict[1]):
                action = 1
            else:
                action = 2
        return action
    
    def memorize(self, state, action, reward, new_state, done):
        if(done == 1):
            self.memory.append((state, action, reward, new_state, True))
        else:
            self.memory.append((state, action, reward, new_state, False))
        
    def replay(self):
        print("replay buffer started")
        print("memory size: {}".format(len(self.memory)))
        size = 200
        batch = random.sample(self.memory, size)
        
        for state, action, reward, new_state, done in batch:
            
            #print("replay")
            # formula when reaching end goal in this timestep:
            target = reward
            # formula when not reaching end goal in this timestep:
            if not done:
                data = np.expand_dims(np.swapaxes(np.array(self.state2image(new_state), dtype='d'), 0, 2), axis=0)
                x = torch.tensor(data, dtype=torch.double)
                Q_target_output = self.model.forward(x)
                
                # real reward
                target = reward + self.gamma*torch.max(Q_target_output)
                
            # the prediction
            data = np.expand_dims(np.swapaxes(np.array(self.state2image(state), dtype='d'), 0, 2), axis=0)
            x = torch.tensor(data, dtype=torch.double)
            Q_predicted_output = self.model.forward(x)
            prediction = Q_predicted_output[action-1]
            
            print(target)
            loss = (prediction-(target.item())).pow(2)
            loss.backward()
        if self.eps_min < self.epsilon:
            self.epsilon -= self.eps_decay
        print("replay buffer ended")
        print("current epsilon: {}".format(self.epsilon))
            
        
    
class Environment:
    def __init__(self, ip = "127.0.0.1", port = 13000, size = 150, timescale = 8):
        self.client     = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip         = ip
        self.port       = port
        self.size       = size
        self.timescale  = timescale

        self.client.connect((ip, port))

    def reset(self):
        self._send(1, 0)
        return self._receive()

    def step(self, action):
        self._send(2, action)
        return self._receive()

    def state2image(self, state):
        return Image.fromarray(np.array(state, "uint8").reshape(self.size, self.size, 3))

    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data   = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end    = data[0]
        reward = data[1]
        state  = [data[i] for i in range(2, len(data))]

        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))
        
        

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=4, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=1)
        
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        
        # Inputs to hidden layer linear transformation
        self.dense1 = nn.Linear(32*17*17, 20)
        self.dense2 = nn.Linear(20, 10)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(10, 2)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.dense1(x.flatten()))
        x = F.relu(self.dense2(x))
        x = self.output(x)
        return x
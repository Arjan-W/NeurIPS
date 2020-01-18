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
import copy
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Agent:
    def __init__(self):
        self.model = Network().double()
        self.target_model = copy.deepcopy(self.model)
        self.replay_count = 0
        self.memory = deque(maxlen=1000)
        self.iteration = 0
        
        self.epsilon = 1
        self.eps_decay = 0.01
        self.eps_min = 0.09
        self.optimizer = optim.Adam(self.model.parameters())
        
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
        
    def replay(self, device):
        if len(self.memory) > 500:
            print("replay buffer started")
            print("memory size: {}".format(len(self.memory)))
            size = 20
            batch = random.sample(self.memory, size)
            
            for state, action, reward, new_state, done in self.memory:
                if reward == 10:
                    batch.append((state, action, reward, new_state, done))
                    
            # make copy of Q; this one is kept stable, the original Q (network) is updated.
            self.optimizer.zero_grad()
            loss_list = []
            self.replay_count +=1
            
            
            # mean squared error
            print(datetime.datetime.now())
            print("start calculating error")
            for state, action, reward, new_state, done in batch:
                loss_list.append(self.loss(state, action, reward, new_state, done, device))
            MSE = sum(loss_list)/len(loss_list)
            print("end calculating error")
            print(datetime.datetime.now())
            print("start learning")
            MSE.backward()
            self.optimizer.step()
            print("end learning")
            print(datetime.datetime.now())
        
        
            print("replay buffer ended")
            print("current epsilon: {}".format(self.epsilon))
            
            if self.replay_count % 8 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            
            
    def loss(self, state, action, reward, new_state, done, device="cpu"):
        reward_gpu = torch.tensor(reward).to(device)
        action_gpu = torch.tensor(action).to(device)
        #print("replay")
            # formula when reaching end goal in this timestep:
        target = reward
        # formula when not reaching end goal in this timestep:
        if not done:
            data = np.expand_dims(np.swapaxes(np.array(self.state2image(new_state), dtype='d'), 0, 2), axis=0)
            x = torch.tensor(data, dtype=torch.double).to(device)
            Q_target_output = self.target_model.forward(x)
                
            # real reward
            target = reward_gpu + self.gamma*torch.max(Q_target_output)
                
        # the prediction
        data = np.expand_dims(np.swapaxes(np.array(self.state2image(state), dtype='d'), 0, 2), axis=0)
        x = torch.tensor(data, dtype=torch.double).to(device)
        Q_predicted_output = self.model.forward(x)
        prediction = Q_predicted_output[action_gpu-1]
            
        loss = (prediction-target).pow(2)
        return loss
            
        
    
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
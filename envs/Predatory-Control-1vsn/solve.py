import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter
from typing import Optional
import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PredatoryControl import *

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


env = PredatoryControl(len=1000,r_=100,move_=True,vmax=500,fmax=0.85,max_steps=450)
state = env.reset()

net =  PGN(env.observation_space.shape[0], env.action_space.n)
print(env.action_space.n)
dict_model = net.load_state_dict( torch.load("./Good") )

while True:
        env.render()
        logics = net(torch.FloatTensor(state))

        probs_v = F.softmax(logics, dim=-1)

        a = torch.multinomial(probs_v, num_samples=1).item()

        state, reward, is_done, _ = env.step(a)

        if is_done:
            state = env.reset()

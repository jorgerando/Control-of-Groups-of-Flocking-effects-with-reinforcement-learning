import gym
import numpy as np
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from FollowMe import *

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

env = followMe(len=1000,n=10,r_=50,change_=True,block=True,n_actions=4,max_steps=5000)
state = env.reset()

net =  PGN(env.observation_space.shape[0], env.action_space.n)
dict_model = net.load_state_dict( torch.load("Good") )

while True:
    env.render()
    logics = net(torch.FloatTensor(state))

    # Aplicar softmax con la dimensión especificada
    probs_v = F.softmax(logics, dim=-1)

    # Muestrear la acción y convertirla a un entero
    a = torch.multinomial(probs_v, num_samples=1).item()

    # Tomar un paso en el entorno
    state, reward, is_done, _ = env.step(a)

    if is_done:
        state = env.reset()

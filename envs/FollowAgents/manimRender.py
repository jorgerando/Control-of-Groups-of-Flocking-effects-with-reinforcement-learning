import gym
#import ptan
import numpy as np

import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from FollowAgents import *
from manim import *

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

# manim -pql -r 3840,2160 manimRender.py My

class My(Scene):

    def construct(self):

        # Crear una flecha
        self.camera.frame_width = 1080*2  # Cambia la anchura a 10 unidades
        self.camera.frame_height = 720*2

        env = followAgents(len=1000 ,n=5,n_actions=5,block=True,random_=False,move=True,max_steps=1000)
        state = env.reset()

        net =  PGN(env.observation_space.shape[0], env.action_space.n)
        dict_model = net.load_state_dict( torch.load("GoodNet") )

        self.camera.frame_center = 1000/2 * RIGHT + 1000/2 * UP

        for i in range(1000) :
                #env.render()
                self.clear()

                env.targets.maninDraw(self,BLUE)
                env.agent.maninDraw(self,vd_=True)

                circle = Circle(radius=560,color=WHITE,stroke_width=250)
                circle.move_to(self.camera.frame_center)
                self.add(circle)

                logics = net(torch.FloatTensor(state))

                probs_v = F.softmax(logics, dim=-1)

                actions = ["→","↑","↓","←","-"]
                chart = BarChart(values=probs_v.tolist(),bar_colors=[BLUE]*5,bar_names=actions , y_range=[0,1,1],).scale(100)
                chart.move_to( [-300 , 950 ,0] )
                self.add(chart)

                a = torch.multinomial(probs_v, num_samples=1).item()

                state, reward, is_done, _ = env.step(a)

                text = Text("Accion : "+str(actions[a]), font_size=36).scale(80)
                text2 = Text("Recompensa : "+str(round(reward,5)), font_size=36).scale(80)

                text.next_to(chart, DOWN*200)
                text2.next_to(text, DOWN*100)
                '''
                stra = ""
                for i in range(5):
                    stra += str( round(state[i*2],3) ) +" "+ str( round(state[i*2+1],3) ) + " \n "

                text3 = Text(stra, font_size=36).scale(80)
                text3.move_to([1300,1000,0])
                '''
                self.add(text,text2)

                if i % 1 == 0 :
                  self.wait(0.1)

                if is_done:
                    break
                    state = env.reset()

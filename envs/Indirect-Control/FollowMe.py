import gym
import numpy as np
import pygame
import random
import gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import *

class followMe(gym.Env):

     def __init__(self , len=10 , r_=25 ,n=5 ,change_=False , block=False ,n_actions=5 ,vmax=10,fmax=0.8 ,max_steps=250 ):

      self.max = max_steps
      self.steps = 0
      self.screen = None

      self.len = len
      self.r = r_
      self.n = n
      self.vmax = vmax
      self.fmax = fmax
      self.n_actions = n_actions
      self.block = block

      angle = random.random() *np.pi * 2
      pi_a = np.array( [ self.len/2 + self.len * 0.35 * np.cos(angle) ,self.len/2+ self.len * 0.35 * np.sin(angle) ] )
      self.agent = Agent(pi_a ,self.vmax  , self.fmax, ind=True  )

      angle = random.random() *np.pi * 2
      pi_a = np.array( [ self.len/2 + self.len * 0.35 * np.cos(angle) ,self.len/2+ self.len * 0.35 * np.sin(angle) ] )

      self.group = Group(p=pi_a,n=n,vmax=self.max/2,r=self.r,color=(0,255,0),block=self.block)
      self.group.addAgent(self.agent)

      self.action_space = gym.spaces.Discrete(self.n_actions)
      self.observation_space = gym.spaces.Box( low=-len , high=len , shape=(6,) ,dtype=int )


      self.point = [self.len/2,self.len/2]

      self.change_time = 500

      self.change = change_

      self.edges = [ [self.len/2,self.len/2] ,
      [self.len/2,self.len*0.25] ,
      [self.len*0.25,self.len/2] ,
      [self.len/2,self.len*0.75] ,
      [self.len*0.75,self.len/2]
      ]

      self.i = 0

     def action(self,action):

           vel_mag = 8
           f = np.array([0.,0.])

           if (self.n_actions == 5 or self.n_actions == 4 ) :
            if action == 0  :
               f = self.agent.seek(np.array([vel_mag,0.]))
            elif action == 1  :
               f = self.agent.seek(np.array([0.,vel_mag]))
            elif action == 2  :
               f = self.agent.seek(np.array([0.,-vel_mag]))
            elif action == 3  :
               f = self.agent.seek(np.array([-vel_mag,0.]))
            elif action == 4 :
               f = self.agent.seek(np.array([0.,0.]))
            elif  action !=0 and  action !=1 and action !=2 and action !=3 and action !=4 :
               print("Invalid Action " +str(action) )
           else :
               if action == 0  :
                  f = self.agent.seek(np.array([vel_mag,0.]))
               elif action == 1  :
                  f = self.agent.seek(np.array([vel_mag,vel_mag]))
               elif action == 2  :
                  f = self.agent.seek(np.array([0.,vel_mag]))
               elif action == 3  :
                  f = self.agent.seek(np.array([-vel_mag,vel_mag]))
               elif action == 4 :
                  f = self.agent.seek(np.array([-vel_mag,0.]))
               elif action == 5 :
                   f = self.agent.seek(np.array([-vel_mag,-vel_mag]))
               elif action == 6 :
                  f = self.agent.seek(np.array([0.,-vel_mag]))
               elif action == 7 :
                  f = self.agent.seek(np.array([vel_mag,-vel_mag]))
               elif action == 8 :
                  f = self.agent.seek(np.array([0.,0.]))
           self.agent.applyForce(f)

     def done(self):
         return self.steps >= self.max

     def observation(self):

          diff = self.agent.getP() - self.group.getMeanP()
          diff_c = np.array( self.point) - self.group.getMeanP()
          diff_ac = np.array( self.point) - self.agent.getP()

          diff /=100
          diff_c /=100
          diff_ac /=100

          return np.array( [diff[0],diff[1],diff_c[0],diff_c[1],diff_ac[0],diff_ac[1]] )

     def renward(self):

           diff = self.agent.getP() - self.group.getMeanP()
           diffc = np.array( self.point ) - self.group.getMeanP()

           if np.linalg.norm( diff ) < self.r + self.r *0.01 :
             return 1+(1/( np.linalg.norm( diffc ) + 1)**2 )*100
           else:
             return 0

     def render(self):
          if self.screen is None:
              pygame.init()
              self.screen = pygame.display.set_mode( (self.len , self.len) )
              self.clock = pygame.time.Clock()

          for event in pygame.event.get():
           if event.type == pygame.QUIT:
             done = True

          pygame.display.set_caption("Indirect control (multi-agent)")

          self.screen.fill((255,255,255))

          diffc =  np.linalg.norm( np.array(self.point) - self.group.getMeanP() )

          pygame.draw.circle( self.screen , (0,0,0) , (self.point[0],self.point[1]) , 5 )
          pygame.draw.circle( self.screen , (0,0,0) , (self.point[0],self.point[1]) , 100/2 ,3 )

          if diffc < 100 :

               pygame.draw.circle( self.screen , (0,255,0) , (self.point[0],self.point[1]) , 5 )
               pygame.draw.circle( self.screen , (0,255,0) , (self.point[0],self.point[1]) , 100/2 ,5 )

          self.group.draw(self.screen)
          self.agent.draw(self.screen,True)

          pygame.display.flip()
          self.clock.tick(60)

     def reset(self):

          angle = random.random() * np.pi * 2
          pi_a = np.array( [ self.len/2 + self.len * 0.45 * np.cos(angle) ,self.len/2+ self.len * 0.45 * np.sin(angle) ] )
          self.agent = Agent(pi_a ,self.vmax  , self.fmax, ind=True  )

          angle = random.random() * np.pi * 2
          pi_a = np.array( [ self.len/2 + self.len * 0.45 * np.cos(angle) ,self.len/2+ self.len * 0.45 * np.sin(angle) ] )

          self.group = Group(p=pi_a,n=self.n,block=self.block,r=self.r,vmax=self.vmax/2,color=(0,255,0))
          self.group.addAgent(self.agent)

          self.steps = 0

          self.i = 0

          return self.observation()

     def step(self,action):

         self.group.flockingMove(self.len)
         self.agent.dodgeWalls_(self.len+self.len*0.10)

         self.action(action)

         info = {}

         if self.change and self.steps % self.change_time == 0 :
              self.point = self.edges[self.i]
              self.i = (self.i + 1) % len(self.edges)

         self.steps+=1

         return self.observation() , self.renward() , self.done() , info

'''
env = followMe(len=1000,n=3)
obs = env.reset()

while True :
    obs , r , d ,_ = env.step(env.action_space.sample())
    env.render()
    if d :
        env.reset()
'''

import gym
import numpy as np
import pygame
import random
import gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import *

class PredatoryControl(gym.Env):

     def __init__(self , len=10 , n_=5 , r_=50 , n_actions=5 ,vmax=10,fmax=0.4 ,move_=False ,max_steps=250 ):

       self.max = max_steps
       self.steps = 0
       self.screen = None

       self.len = len
       self.r = r_
       self.vmax = vmax
       self.fmax = fmax
       self.move =move_

       self.n = n_

       pi_a = np.array( [ random.random() *self.len - self.len*0.1 ,random.random() *self.len - self.len*0.1 ] )
       self.agent = Agent(pi_a ,self.vmax  , self.fmax )

       pi_g = np.array( [  self.len/2 + self.len/4, self.len/2] )
       self.targets = Group( p=pi_g,n=self.n,vmax=5,block=True,color=(0,255,0) )

       self.n_actions = n_actions

       self.action_space = gym.spaces.Discrete(self.n_actions)
       self.observation_space = gym.spaces.Box( low=-len , high=len , shape=(6,) ,dtype=int )

       self.center = self.targets.getMeanP()
       self.max_radius = self.targets.groupRadius()

     def observation(self):

         diff = self.agent.getP() - self.targets.getMeanP()
         diff_c = np.array([self.len/2,self.len/2]) - self.targets.getMeanP()
         diff_ac = np.array([self.len/2,self.len/2]) - self.agent.getP()
         diff_r = self.r - self.max_radius

         diff /=100
         diff_c /=100
         diff_ac /=100
         diff_r /= 100

         return np.array( [diff[0],diff[1],diff_c[0],diff_c[1],diff_ac[0],diff_ac[1]] )

     def done(self):
          return self.steps >= self.max #or self.max_radius > self.r

     def renward(self):

           diff = self.agent.getP() - self.targets.getMeanP()
           diffc = np.array([self.len/2,self.len/2]) - self.targets.getMeanP()

           renw = 0

           if np.linalg.norm( diff ) < self.r + self.r*0.1 :
             return  renw + 1 + (1/( np.linalg.norm( diffc ) + 1)**2)*100
           else:
             return renw

     def action(self,action):

          vel_mag = 8
          f = np.array([0.,0.])
          if (self.n_actions == 5) :
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

     def reset(self):

          pi_a = np.array( [ random.random() *self.len - self.len*0.1  ,random.random() *self.len - self.len*0.1 ] )
          self.agent = Agent(pi_a ,self.vmax, self.fmax)

          pi_g = np.array( [  self.len/2 + self.len/4, self.len/2] )
          self.targets = Group( p=pi_g,n=self.n,vmax=5,block=True,color=(0,255,0) )
          self.steps = 0

          return self.observation()

     def render(self):

         if self.screen is None:
             pygame.init()
             self.screen = pygame.display.set_mode( (self.len , self.len) )
             self.clock = pygame.time.Clock()

         for event in pygame.event.get():
          if event.type == pygame.QUIT:
            done = True

         self.screen.fill((255,255,255))

         pygame.draw.circle( self.screen , (255,0,0) , (self.len/2,self.len/2) , 5 )

         p = self.agent.getP()
         pygame.draw.circle( self.screen , (255,0,0) , (p[0],p[1]) , self.r )

         pygame.draw.circle(self.screen,(0,0,0),(self.center[0],self.center[1]),self.max_radius + 10 ,4)

         #pygame.draw.circle(self.screen,(0,0,0),(self.center[0],self.center[1]),self.max_radius + self.max_radius*0.5 + self.r,4)


         self.targets.draw(self.screen)

         #pygame.draw.circle( self.screen , (0,0,0) , (self.center[0],self.center[1]) , 5 )

         self.agent.draw(self.screen,True)

         pygame.display.flip()
         self.clock.tick(60)

     def step(self,action):

         self.center = self.targets.getMeanP()
         self.max_radius = self.targets.groupRadius()

         self.agent.move()

         if self.move :
            self.targets.flockingMove(self.len,ch=4)

         for target in self.targets.agents :

             #target.v *= 0.5
             if not self.move :
                target.v *= 0.5
                target.move()

                f = target.flee(self.r,self.agent.getP())
                target.applyForce(f*2)
             else:
                 f = target.flee(self.r,self.agent.getP())
                 target.applyForce(f*0.5)

             target.block(self.len)

         #self.target.v *= 0.5
         self.action(action)

         #if self.random :
           #f = self.target.randomMove(self.len)
           #self.target.applyForce(f)

         #f = self.target.flee(self.r,self.agent.getP())
         #self.target.applyForce(f*2)

         #self.target.block(self.len)
         self.agent.block(self.len*2)

         info = {}
         self.steps+=1

         return self.observation() , self.renward() , self.done() , info

'''
env = PredatoryControl(len=500,r_=50,n_actions=5,vmax=50,fmax=0.85)
obs = env.reset()
while True:
    obs , r , d , _ = env.step(env.action_space.sample())
    env.render()
    if d :
        env.reset()
'''

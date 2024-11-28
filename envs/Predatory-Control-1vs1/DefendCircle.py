import gym
import numpy as np
import pygame
import random
import gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import *

class defendCircle(gym.Env):

     def __init__(self , len=10 , r_=50 , n_actions=5 ,vmax=10,fmax=0.4 ,random_=False ,max_steps=250 , change_point=False):

       self.max = max_steps
       self.steps = 0
       self.screen = None

       self.len = len
       self.r = r_
       self.vmax = vmax
       self.fmax = fmax
       self.random =random_
       self.change = change_point
       self.change_time = 350
       self.i = 0

       pi_a = np.array( [ random.random() *self.len- self.len*0.1 ,random.random() *self.len - self.len*0.1 ] )
       self.agent = Agent(pi_a ,self.vmax  , self.fmax )


       pi_g = np.array( [  self.len/2 + self.len/4, self.len/2] )

       self.target = Agent(pi_g ,vmax=5 , fmax=2 )

       self.n_actions = n_actions

       self.action_space = gym.spaces.Discrete(self.n_actions)
       self.observation_space = gym.spaces.Box( low=-len , high=len , shape=(6,) ,dtype=int )

       self.point = [self.len/2,self.len/2]

       self.edges = [ [self.len/2,self.len/2] ,
       [self.len/2,self.len*0.25] ,
       [self.len*0.25,self.len/2] ,
       [self.len/2,self.len*0.75] ,
       [self.len*0.75,self.len/2]
       ]

     def observation(self):
         diff = self.agent.getP() - self.target.getP()
         diff_c = np.array(self.point) - self.target.getP()
         diff_ac = np.array(self.point) - self.agent.getP()

         diff /=100
         diff_c /=100
         diff_ac /=100

         return np.array( [diff[0],diff[1],diff_c[0],diff_c[1],diff_ac[0],diff_ac[1]] )

     def done(self):
          return self.steps >= self.max

     def renward(self):

           diff = self.agent.getP() - self.target.getP()
           diffc = np.array(self.point) - self.target.getP()

           if np.linalg.norm( diff ) < self.r + self.r * 0.1 :
             return  1 + (1/( np.linalg.norm( diffc ) + 1)**2)*100
           else:
             return 0

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

          pi_a = np.array( [ random.random() *self.len - self.len*0.1 ,random.random() *self.len - self.len*0.1 ] )
          self.agent = Agent(pi_a ,self.vmax, self.fmax)

          pi_g = np.array( random.choice(self.edges[1:]) )
          #print(pi_g)

          self.target = Agent(pi_g ,vmax=2.5 , fmax=5.5 ,color=(0,255,0))
          self.steps = 0
          self.i = 0

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

         pygame.display.set_caption("Predator Control (Single Agent) ")


         diffc =  np.linalg.norm( np.array(self.point) - self.target.getP() )


         p = self.agent.getP()
         pygame.draw.circle( self.screen , (255,0,0) , (p[0],p[1]) , self.r )

         self.target.draw(self.screen)

         self.agent.draw(self.screen,True)

         pygame.draw.circle( self.screen , (0,0,0) , (self.point[0],self.point[1]) , 5 )
         pygame.draw.circle( self.screen , (0,0,0) , (self.point[0],self.point[1]) , self.r/2 ,3 )

         if diffc < self.r/2 :
              #dibujar un circulo transparente >:/
              '''superficie_circulo = pygame.Surface((self.r,self.r), pygame.SRCALPHA)
              pygame.draw.circle(superficie_circulo, (0,255,0,100), (self.r/2,self.r/2), self.r/2)
              self.screen.blit(superficie_circulo,(self.point[0]-self.r/2,self.point[1]-self.r/2)  )
              '''
              pygame.draw.circle( self.screen , (0,255,0) , (self.point[0],self.point[1]) , 5 )
              pygame.draw.circle( self.screen , (0,255,0) , (self.point[0],self.point[1]) , self.r/2 ,5 )

         pygame.display.flip()
         self.clock.tick(60)

     def step(self,action):

         self.agent.move()
         self.target.move()
         self.target.v *= 0.5
         self.action(action)

         if self.random :
           f = self.target.randomMove(self.len)
           self.target.applyForce(f)

         f = self.target.flee(self.r,self.agent.getP())
         self.target.applyForce(f*2)

         self.target.block(self.len)
         self.agent.block(self.len + self.len*0.25)

         info = {}

         if self.change and self.steps % self.change_time == 0 :
              self.point = self.edges[self.i]
              self.i = (self.i + 1) % len(self.edges)

         self.steps+=1

         return self.observation() , self.renward() , self.done() , info

'''
env = defendCircle(len=500 , r_= 150 )
obs = env.reset()
while True:
    obs , r , d , _ = env.step(1)
    env.render()
    if d :
        env.reset()
'''

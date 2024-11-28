import gym
import numpy as np
import pygame
import random
import gym
import sys

# simple vector operations

def limit(f,limit):
    mag = np.linalg.norm(f)
    if mag > limit :
        return f * limit / mag
    else :
        return f

def setMag(v,new_mag):
    mag = np.linalg.norm(v)
    if mag == 0 :
        return v
    uni = v / mag
    uni *= new_mag
    return uni

def dist_(a,b):
    sub = a - b
    return np.linalg.norm(sub)

def normalize(a):
    mag = np.linalg.norm(a)
    if mag == 0 :
        return a
    uni = a / mag
    return uni

# Flocking algoritm implementation 

class Agent :

     def __init__(self , p_i , vmax=7 , fmax=0.1 , r=25 , color=(80,80,80) ,ind=False) :

         self.p = p_i.astype(float);

         self.v = np.array([0.,0.])
         self.a = np.array([0.,0.])
         self.vd =np.array([0.,0.])

         self.vmax = vmax
         self.fmax = fmax
         self.r = r  # vision range
         self.ind = ind

         self.color = color

         #random rd_move
         self.rd_angl = random.random()*np.pi*2

     def getP(self):
         return self.p.copy()

     def move(self):
         self.v += self.a
         self.p += self.v
         self.a *= 0.

     def applyForce(self,f):
         self.a += f

     def dir(self,vd):

          self.vd = vd
          dif = self.vd - self.v
          mag = np.linalg.norm(dif)

          if mag == 0 :
              f = np.array([0.,0.])
          else :
              f = dif / mag
              f *= 0.25
          return f

     def seek(self,vd):
         #self.vd = vd
         steer = vd - self.v.copy()
         self.vd = vd
         f = limit(steer,self.fmax)
         return f

     def seek_td(self,td):
         desired = td - self.p
         vd = setMag(desired,self.vmax)
         return self.seek(vd)

     def block(self,len):
         if self.p[0] > len :
            self.p[0] = len
         elif self.p[0] < 0 :
            self.p[0] = 0

         if self.p[1] > len :
            self.p[1] = len
         elif self.p[1] < 0 :
            self.p[1] = 0

     def randomMove(self,len):

        rd = 2.5
        rd_angle_inc = np.pi/2.5

        v_x = setMag(self.v,rd)
        c = v_x + self.p

        self.rd_angl += random.uniform(-rd_angle_inc, rd_angle_inc)
        td  = np.array( [ c[0] + 7 * np.cos(self.rd_angl) , c[1] + 7 * np.sin(self.rd_angl) ] )


        self.block(len)

        if self.isNear2Wall(len):
            return self.dodgeWalls_(len)
        else :
            return self.seek_td(td)

     def flee(self,r,p):

         f = np.array([0.,0.])
         diff = p - self.p
         mag = np.linalg.norm(diff)

         if mag < r :
             desired = self.p - p
             vd = setMag(desired,self.vmax*1.5)
             self.vd = vd
             steer = vd - self.v.copy()
             f = steer

         return f

     def separation(self,others):

         sum = np.array([0.,0.])
         f = np.array([0.,0.])
         count = 0

         for other in others:
             dist = dist_(other.getP(),self.p)
             if dist < 25 and dist > 0 :
                  diff = self.p - other.getP()
                  uni = normalize(diff)
                  uni /= dist
                  sum += uni
                  count += 1

         if count > 0 :
            sum /= count
            vd = setMag(sum,self.vmax)
            steer = vd - self.v
            f = limit(steer,self.fmax)

         return f

     def cohesion(self,others):
          sum = np.array([0.,0.])
          f = np.array([0.,0.])
          count = 0

          for other in others:
              dist = dist_(other.getP(),self.p)
              if dist < 75 and dist > 0 :
                  sum += other.p
                  count+=1

          if count > 0 :
             p_m = sum / count
             desired = p_m - self.p
             vd = setMag(desired,self.vmax)
             self.vd = vd
             steer = vd - self.v
             f = limit(steer,self.fmax)
          return f

     def aling(self,others):
          sum = np.array([0.,0.])
          f = np.array([0.,0.])
          count = 0

          for other in others:
              dist = dist_(other.getP(),self.p)
              if dist < 50 and dist > 0 :
                  sum += other.v
                  count += 1

          if count > 0 :
              sum /= count
              vd = setMag(sum,self.vmax)

              steer = vd - self.v
              f = limit(vd,self.fmax)
          return f

     def flockingForces(self,others):

         f_aling = np.array([0.,0.])
         sum_aling = np.array([0.,0.])

         f_cohesion = np.array([0.,0.])
         sum_cohesion = np.array([0.,0.])

         f_separation = np.array([0.,0.])
         sum_separation = np.array([0.,0.])

         count_separation = 0
         count_aling = 0
         count_cohesion = 0

         if self.ind :
            return f_aling , f_cohesion , f_separation

         for other in others:

             dist = dist_(other.getP(),self.p)

             #separation
             if dist < self.r and dist > 0 :
                  diff = self.p - other.getP()
                  uni = normalize(diff)
                  uni /= dist
                  sum_separation += uni
                  count_separation += 1

             #cohesion
             if dist < self.r*3 and dist > 0 :
                 sum_cohesion += other.p
                 count_cohesion+=1

             #aling
             if dist < self.r*2 and dist > 0 :
                a = 1
                if other.ind:
                    a = 1000.
                sum_aling += other.v #* a
                count_aling += 1

         #separation
         if count_separation > 0 :
           sum_separation /= count_separation
           vd = setMag(sum_separation,self.vmax)
           steer = vd - self.v
           f_separation = limit(steer,self.fmax)

         #cohesion
         if count_cohesion > 0 :
            p_m = sum_cohesion / count_cohesion
            desired = p_m - self.p
            vd = setMag(desired,self.vmax)
            self.vd = vd
            steer = vd - self.v
            f_cohesion = limit(steer,self.fmax)

         #aling
         if count_aling > 0 :
             sum_aling /= count_aling
             vd = setMag(sum_aling,self.vmax)
             steer = vd - self.v
             f_aling = limit(steer,self.fmax)

         return f_aling , f_cohesion , f_separation

     def flockingForcesPonderate(self,others):
         # tiene un poco mas de influencia sobre el grupo
         p = 1
         inportancia = 1/2.5 # de 0 a 1

         f_aling = np.array([0.,0.])
         sum_aling = np.array([0.,0.])
         sum_aling_ind = np.array([0.,0.])

         f_cohesion = np.array([0.,0.])
         sum_cohesion = np.array([0.,0.])
         sum_cohesion_ind = np.array([0.,0.])

         f_separation = np.array([0.,0.])
         sum_separation = np.array([0.,0.])
         sum_separation_ind = np.array([0.,0.])

         count_separation = 0
         count_separation_ind = 0

         count_aling = 0
         count_aling_ind = 0

         count_cohesion = 0
         count_cohesion_ind = 0

         if self.ind :
            return f_aling , f_cohesion , f_separation

         for other in others:

             dist = dist_(other.getP(),self.p)

             #separation
             if dist < self.r and dist > 0 :
                  diff = self.p - other.getP()
                  uni = normalize(diff)
                  uni /= dist

                  if other.ind :
                    sum_separation_ind += uni
                    count_separation_ind += 1
                  else :
                    sum_separation += uni
                    count_separation += 1

             #cohesion
             if dist < self.r*3 and dist > 0 :
                 if other.ind :
                     sum_cohesion_ind += other.p
                     count_cohesion_ind += 1
                 else :
                     sum_cohesion += other.p
                     count_cohesion += 1

             #aling
             if dist < self.r*2 and dist > 0 :
                if other.ind :
                    sum_aling_ind += other.v
                    count_aling_ind += 1
                else :
                    sum_aling += other.v
                    count_aling += 1

         #separation
         steer = 0
         steer_ind = 0

         if count_separation > 0 :
             sum_separation /= count_separation
             vd = setMag(sum_separation,self.vmax  )
             steer = vd - self.v

         if count_separation_ind > 0 :
             sum_separation_ind /= count_separation_ind
             vd_ind = setMag(sum_separation_ind,self.vmax * p )
             steer_ind = vd_ind - self.v

         steer = steer *2/3 + steer_ind * 1/3
         f_separation = limit(steer,self.fmax)

         #cohesion
         steer = 0
         steer_ind = 0

         if count_cohesion > 0 :
            p_m = sum_cohesion / count_cohesion
            desired = p_m - self.p
            vd = setMag(desired,self.vmax )
            steer = vd - self.v

         if count_cohesion_ind > 0 :
            p_m_ind = sum_cohesion_ind / count_cohesion_ind
            desired_ind = p_m_ind - self.p
            vd_ind = setMag(desired_ind,self.vmax * p )
            steer_ind = vd_ind - self.v

         steer = steer*2/3 + steer_ind*1/3
         f_cohesion = limit(steer,self.fmax)

         #aling
         steer = 0
         steer_ind = 0

         if count_aling > 0 :
             sum_aling /= count_aling
             vd = setMag(sum_aling,self.vmax )
             steer = vd - self.v

         if count_aling_ind > 0 :
             sum_aling_ind /= count_aling_ind
             vd_ind = setMag(sum_aling_ind,self.vmax * p)
             steer_ind = vd_ind - self.v*2

         steer = steer * 2/3 + steer_ind * 1/3
         f_aling = limit(steer,self.fmax)

         return f_aling , f_cohesion , f_separation


     def dodgeWalls(self,len):

         if self.p[0] > len :
            self.p[0] = 0
         elif self.p[0] <  0 :
            self.p[0] = len
         elif self.p[1] > len :
            self.p[1] = 0
         elif self.p[1] < 0:
            self.p[1] = len

     def dodgeWalls_(self,len):

          p_ = self.p.copy()

          diff = p_ - np.array([len/2,len/2])
          diff *= -random.random()
          vd = setMag(diff,self.vmax)

          return self.seek(vd)

     def isNear2Wall(self,len):
         diff = self.p - np.array([len/2,len/2])
         mag = np.linalg.norm(diff)
         return mag > (len/2 -len*0.01)

     def draw(self,screen,draw_vs=False):

         v1 = np.array([-10.,-15.])
         v2 = np.array([10.,-15.])
         v3 = np.array([0., 20.])
         vertexs = [v1,v2,v3]

         # v angle
         alfa = np.arctan2(self.v[1], self.v[0]) - np.pi/2

         for v in vertexs :
             # rotate
             vx = v[0]
             vy = v[1]
             v[0] = np.cos(alfa)*vx - np.sin(alfa)*vy
             v[1] = np.sin(alfa)*vx + np.cos(alfa)*vy
             #translate
             v += self.p

         pygame.draw.polygon(screen,self.color,vertexs)

         #draw vel
         mult = 5
         if draw_vs:
          pygame.draw.line(screen,(255,0,0),self.p,(self.p[0]+self.v[0]*mult,self.p[1]+self.v[1]*mult),3)
          pygame.draw.line(screen,(0,255,0),self.p,(self.p[0]+self.vd[0]*mult,self.p[1]+self.vd[1]*mult),3)

     def maninDraw(self,scene,color='#888888',vd_=False):

          #nomalizo las cordenadas a la escena de manim
          cx = self.p[0]
          cy = self.p[1]
          vs = [ np.array([-10.,-15.]),np.array([10.,-15.]),np.array([0.,20.])]
          v_an = np.arctan2(self.v[1], self.v[0]) - np.pi/2

          for v in vs :
              x = v[0]
              y = v[1]
              v[0] = np.cos(v_an)*x - np.sin(v_an)*y
              v[1] = np.sin(v_an)*x + np.cos(v_an)*y
              v += self.p

          # Crear el triángulo usando los vértices definidos
          import manim

          triangle = manim.Polygon([vs[0][0],vs[0][1],0],[vs[1][0],vs[1][1],0],[vs[2][0],vs[2][1],0],fill_color=color,fill_opacity=1,stroke_color=manim.WHITE,stroke_width=150)
          scene.add(triangle.round_corners(radius=0.5))
          mult = 7

          if vd_ :
           vd = manim.Arrow(start=[cx,cy,0], end=[cx+self.vd[0]*mult,cy+self.vd[1]*mult,0], color=manim.PURE_GREEN, stroke_width=550)
           scene.add(vd)

           v = manim.Arrow(start=[cx,cy,0], end=[cx+self.v[0]*mult,cy+self.v[1]*mult,0], color=manim.PURE_RED, stroke_width=550)
           scene.add(v)

class Group :

    def __init__ (self , p=np.array([0,0]) ,block=False, vmax=7 ,incre=10 , fmax=0.2 , r=20, n=5 ,color=(80,80,80) ) :

        self.n = n
        self.agents = []
        self.block = block

        for _ in range(self.n):
            p_i = p + np.array([random.randint(-incre,incre),random.randint(-incre,incre)])
            self.agents.append(Agent(p_i=p_i,vmax=vmax,fmax=fmax,r=r,color=color))

    def getAgents(self):
        return self.agents

    def addAgent(self,agent):
        self.agents.append(agent)
        self.n+=1

    def getMeanP(self):
        sum = np.array([0.,0.])
        n = 0
        for a in self.agents :
            if a.ind :
                continue
            sum += a.p
            n+=1

        sum /= n
        return sum

    def randomMove(self,size):
        for a in self.agents:
                a.move()
                r = a.randomMove(size)
                a.applyForce(r)

    def flockingMove(self,size,sp=4,ch=1.2,al=1):

        for a in self.agents:

                a.move()

                if a.ind :
                    continue

                if not a.ind:
                  a.block(size)

                f_aling , f_cohesion , f_separation = a.flockingForcesPonderate(self.agents)

                f = f_separation * sp
                a.applyForce(f)
                f2 = f_cohesion * ch
                a.applyForce(f2)
                f3 = f_aling * al
                a.applyForce(f3)

                if a.isNear2Wall(size) and self.block :
                  d = a.dodgeWalls_(size) * 5
                  a.applyForce(d)

    def flockingMove2(self,size,sp=2,ch=1.2,al=1):

        for a in self.agents:

                a.move()

                if not a.ind:
                  a.block(size)

                f_aling , f_cohesion , f_separation = a.flockingForcesPonderate(self.agents)

                f = f_separation * sp
                a.applyForce(f)
                f2 = f_cohesion * ch
                a.applyForce(f2)

                if a.isNear2Wall(size) and self.block :
                  d = a.dodgeWalls_(size) * 5
                  a.applyForce(d)

    def groupRadius(self):

        center = self.getMeanP()
        max = -1

        for a in self.agents :
            diff = a.getP() - center
            mag = np.linalg.norm(diff)
            if mag > max :
                max = mag
        return max

    def groupRadiusMaxMean(self):

        center = self.getMeanP()
        max = -1
        mean = 0

        for a in self.agents :
            diff = a.getP() - center
            mag = np.linalg.norm(diff)
            mean += mag
            if mag > max :
                max = mag

        return max , (mean / self.n)


    def draw(self,screen):
        for a in self.agents:
            a.draw(screen)

    def maninDraw(self,screen,color='#888888'):

        for a in self.agents:
            a.maninDraw(screen,color)

'''
pygame.init()
screen = pygame.display.set_mode( (2000, 2000) )
pygame.display.set_caption("Pruebas")
clock = pygame.time.Clock()

agents = Group(np.array([500,200]),n=10)
#agent = Agent(np.array([500,200]))

while True :

   for event in pygame.event.get():
     if event.type == pygame.QUIT:
       done = True

   screen.fill((255,255,255))
   agents.draw(screen)
   agents.flockingMove(2000)

   pygame.display.flip()
   clock.tick(60)
'''

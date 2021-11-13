import numpy as np
import math
# import assign_pairs
import random
import pygame
import time

# global variables
n = 4 # number of pursuers (0 to n-1)
m = 4 # number of evaders (0 to m-1)


class Evader():
    def __init__(self, x, y, id, size=10):
        self.x = x
        self.y = y
        self.size = size # px
        self.colour = (0, 0, 255)
        self.thickness = 1 # px
        self.speed = 2
        self.ID = id

    ## Helper function to visualize 
    def display(self, screen, scale, w):
        pygame.draw.circle(screen, self.colour, (int(scale * self.x + w / 2), int(-scale * self.y + w / 2)), self.size, self.thickness)

    ## Compute velocity (circular motion)
    def vel(self):
        a = np.arctan2(self.y, self.x)
        vx = -self.speed * np.sin(a)
        vy = self.speed * np.cos(a)
        return vx, vy

    ## Evader movement defined by kinematic model
    def move(self, vx, vy, dt):
        self.x += dt * vx
        self.y += dt * vy


class Pursuer():
    def __init__(self, x, y, size=10):
        self.x = x
        self.y = y
        self.size = size
        self.colour = (255, 0, 0)
        self.thickness = 1
        self.capturing_radius = 10
        self.coordination_radius = 10
        self.I_a = list(range(m))
        self.I_t = []
        self.C_i = []
        self.neighbours_list = []

    ## Helper function to visualize 
    def display(self, screen, scale, w):
        pygame.draw.circle(screen, self.colour, (int(scale * self.x + w / 2), int(-scale * self.y + w / 2)), self.size, self.thickness)

    ## Compute velocity (using strategy, position (ex, ey) of assigned evader)
    def vel(self, ex, ey, t, t0):
        a = 0.25 # have to tune these (see Thm 1, Zavlanos and Pappas 2007)
        R = 4.0
        K = 2.0
        gamma = math.sqrt((self.x - ex) ** 2 + (self.y - ey) ** 2)
        r = R * math.exp(-a * (t - t0))
        beta = r ** 2 - gamma ** 2
        vx = - K * (1 / beta ** 2) * 2 * (self.x - ex)
        vy = - K * (1 / beta ** 2) * 2 * (self.y - ey)
        return vx, vy

    ## Pursuer movement defined by kinematic model
    def move(self, vx, vy, dt):
        self.x += dt * vx
        self.y += dt * vy
        
    # check if the pursuer can capture assigned evader
    def is_within_reach(self, ex, ey): 
        # tune the parameter r 
        return (self.x - ex)**2 + (self.y - ey)**2 <= self.capturing_radius^2 #*exp(-a*(t-t0)))^2

    # Neighbours to achieve local coordination among the pursuers
    def is_neighbor(self, px, py):
        return (self.x - px)**2 + (self.y - py)**2 <= self.coordination_radius #*exp(-a*(t-t0)))^2

    def is_assigned(self):
        return len(self.I_a) == 1

    def evader_assigned(self):
        return self.I_a

    def neighbours(self, list_of_pursuers):
        self.neighbours_list = []
        for pi in list_of_pursuers:
            if self.is_neighbor(pi.x,pi.y):
                self.neighbours_list.append(pi)
        return self.neighbours_list


#This is to generate some velocity field
def vel_form(centers, dim, list_):
    u, v = np.zeros((dim, dim)), np.zeros((dim, dim))
    x, y = np.meshgrid(np.linspace(0,dim,dim+1),np.linspace(0,dim,dim+1))
    for i in range(len(centers)):
        c1, c2 = centers[i]
        cen = (7, 7)
        u1 = -(x-c1)/((x-c1)**2 + (y-c2)**2)**(32/4)
        u1[c2, c1] = 0
        v1 = -(y-c2)/((x-c1)**2 + (y-c2)**2)**(32/4)
        v1[c2, c1] = 0
        u, v = u + list_[i]*u1[1:, 1:].copy(), v + list_[i]*v1[1:, 1:].copy()
    return u, v
    
def min_dist(evader, list_of_pursuers): 
    min_dist = 100000
    s_i = list_of_pursuers[0]
    for pursuer in list_of_pursuers:
        dist = (pursuer.x - evader.x)**2 + (pursuer.y- evader.y)**2
        if (dist < min_dist):
            min_dist = dist
            s_i = pursuer
    return s_i

def task_assignment(P,E):

    e2a = [] # evaders that satisfy assumption 2a
    p2a = []
    for p_ind in range(n):
        for e_ind in range(m):
            if (P[p_ind].is_within_reach(E[e_ind].x, E[e_ind].y)) & (len(P[p_ind].I_a) > 1) & (E[e_ind].ID in P[p_ind].I_a):
                e2a.append(E[e_ind])
                
        if (len(e2a) >= 1):
            # select an evader from all evaders that satisfy assumption 2a, this also incorporates tie breaking 
            random_index = random.randint(0,len(e2a)-1)
            selected_evader = e2a[random_index]
            
            # identify candidate purusers 
            for p_ind in range(n):
                if (P[p_ind].is_within_reach(selected_evader.x, selected_evader.y)) & (len(P[p_ind].I_a) > 1) & (selected_evader.ID in P[p_ind].I_a):
                    p2a.append(P[p_ind])

            # select the closest pursuer 
            selected_pursuer = min_dist(selected_evader, p2a)

            #if selected evader is avaialble, update Ia and It of all neighbouring purusers
            if selected_evader.ID in selected_pursuer.I_a: 
                selected_pursuer.I_a = []
                selected_pursuer.I_a.append(selected_evader.ID)
                selected_pursuer.I_t.append(selected_evader.ID) 
                for neighbour in selected_pursuer.neighbours(P):
                    if len(neighbour.I_a) > 1 :
                        neighbour.I_a.remove(selected_evader.ID)
                    if selected_evader.ID not in neighbour.I_t:
                        neighbour.I_t.append(selected_evader.ID) 

            e2a = []
            p2a = []


def play_game():
    print("Playing...")

    # Set up display
    # https://www.pygame.org/docs/tut/PygameIntro.html
    pygame.init()
    size = width, height = 600, 600 # display size 500 x 500 px
    screen = pygame.display.set_mode(size)
    scale = width / 6
    clock = pygame.time.Clock()
    
    # Wait 3 seconds (for screen recording)
    time.sleep(3)

    # list of initial positions (X = pursuers, Y = evaders)
    # X = # [tuple(float, float), ...]
    # Y = # [tuple(float, float), ...]

    # Initialize pursuers, evaders
    # Field is 6m x 6m, centered at the origin
    # Pursuers start in box [-2, -2] to [-1, -1]
    # Evaders start on a circle of radius R
    P = []
    for ii in range(n):
        x = random.random() - 2.0
        y = random.random() - 2.0
        p = Pursuer(x, y)
        P.append(p)

    E = []
    R = 2.0
    for ii in range(m):
        a = (2 * math.pi / m) * ii
        x = R * np.cos(a)
        y = R * np.sin(a)
        e = Evader(x, y, ii)
        E.append(e)

    dt = 0.01
    
    is_game_over = False
    
    t0 = 0.0
    t = t0
    while not is_game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Assign pursuer-evader pairs
        # Output: 
        # A = [(int: P, int: E), ...]
        # I = list of pairs [(I^a(list: int), I^t(list: int)), ...]
        # Ex:
        # I[0][0]: pursuer 0's I^a list
        # I[4][1]: pursuer 4's I^t list 
        #A = [(0,0), (1,1), (2,2), (3,3)] # hard-coded assignments for now

        task_assignment(P,E)
        A = []
        for p_ind in range(n):
            if len(P[p_ind].I_a) == 1:
                A.append((p_ind, P[p_ind].I_a[0]))
        ii =  1
        for pursuer in P:
            print("Pursuer ", ii ," location: ", pursuer.x, pursuer.y, "Pursuer Assignment",pursuer.I_a, pursuer.I_t )
            ii = ii + 1
        ii =  1
        for evader in E:
            print("Evader ", ii ," location: ", evader.x, evader.y, "Pursuer Assignment",evader.ID)
            ii = ii + 1

        # Integrate dynamics
        for p_ind, e_ind in A:
            e = E[e_ind]
            vx, vy = P[p_ind].vel(e.x, e.y, t, t0)
            P[p_ind].move(vx, vy, dt)

        for ii in range(m):
            vx, vy = E[ii].vel()
            E[ii].move(vx, vy, dt)

        # Visualize
        screen.fill((255,255,255))
        [p.display(screen, scale, width) for p in P]
        [e.display(screen, scale, width) for e in E]
        clock.tick(30)
        pygame.display.update()
        
        # Current time
        t = t + dt

        
        
    

if __name__ == "__main__":
	play_game()
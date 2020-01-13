#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from utils import compute_path_from_wp
from cvxpy_mpc import optimize

import sys
import time

# classes
class MPC():

    def __init__(self):

        # State for the robot mathematical model [x,y,heading]
        self.state =  np.zeros(3)
        # Sim step
        self.dt = 0.25

        # starting guess output
        N = 5 #number of state variables
        M = 2 #number of control variables
        T = 20 #Prediction Horizon
        self.opt_u =  np.zeros((M,T))
        self.opt_u[0,:] = 1 #m/s
        self.opt_u[1,:] = np.radians(0) #rad/s

        # Interpolated Path to follow given waypoints
        self.path = compute_path_from_wp([0,20,30,30],[0,0,10,20])

        #Initialise plot
        # First set up the figure, the axis, and the plot element we want to animate
        plt.style.use("ggplot")
        self.fig = plt.figure()
        plt.ion()
        plt.show()

    def run(self):
        '''
        '''

        while 1:
            if self.state is not None:

                #optimization loop
                start=time.time()
                self.opt_u = optimize(self.state,
                                        self.opt_u,
                                        self.path)

                print("CVXPY Optimization Time: {:.4f}s".format(time.time()-start))

                self.update_sim(self.opt_u[0,1],self.opt_u[1,1])

                self.plot_sim()

    def update_sim(self,lin_v,ang_v):
        '''
        Updates state.

        :param lin_v: float
        :param ang_v: float
        '''

        self.state[0] = self.state[0] +lin_v*np.cos(self.state[2])*self.dt
        self.state[1] = self.state[1] +lin_v*np.sin(self.state[2])*self.dt
        self.state[2] = self.state[2] +ang_v*self.dt

    def plot_sim(self):
        plt.clf()
        self.ax = plt.axes(xlim=(np.min(self.path[0,:])-1, np.max(self.path[0,:])+1),
                      ylim=(np.min(self.path[1,:])-1, np.max(self.path[1,:])+1))

        self.track, = self.ax.plot(self.path[0,:],self.path[1,:], "g-", label="reference track")
        self.vehicle, = self.ax.plot([self.state[0]], [self.state[1]], "r*", label="vehicle path")
        plt.legend()
        plt.draw()
        plt.pause(0.1)

def do_sim():
    sim=MPC()
    try:
        sim.run()
    except Exception as e:
        sys.exit(e)

if __name__ == '__main__':
    do_sim()

#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from utils import compute_path_from_wp
from cvxpy_mpc import optimize

import sys
import time

# Robot Starting position
SIM_START_X=0
SIM_START_Y=2
SIM_START_H=0

# Classes
class MPC():

    def __init__(self):

        # State for the robot mathematical model [x,y,heading]
        self.state =  [SIM_START_X, SIM_START_Y, SIM_START_H]
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

        # Sim help vars
        self.sim_time=0
        self.x_history=[]
        self.y_history=[]
        self.h_history=[]
        self.v_history=[]
        self.w_history=[]

        #Initialise plot
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

        self.sim_time = self.sim_time+self.dt
        self.x_history.append(self.state[0])
        self.y_history.append(self.state[1])
        self.h_history.append(self.state[2])
        self.v_history.append(self.opt_u[0,1])
        self.w_history.append(self.opt_u[1,1])

        plt.clf()

        grid = plt.GridSpec(2, 3)

        plt.subplot(grid[0:2, 0:2])
        plt.title("MPC Simulation \n" + "Simulation elapsed time {}s".format(self.sim_time))

        plt.plot(self.path[0,:],self.path[1,:], c='tab:orange', marker=".", label="reference track")
        plt.plot(self.x_history, self.y_history, c='tab:blue', marker=".", alpha=0.5, label="vehicle trajectory")
        plt.plot(self.x_history[-1], self.y_history[-1], c='tab:blue', marker=".", markersize=12, label="vehicle position")
        plt.arrow(self.x_history[-1], self.y_history[-1],np.cos(self.h_history[-1]),np.sin(self.h_history[-1]),color='tab:blue',width=0.2,head_length=0.5)
        plt.ylabel('map y')
        plt.xlabel('map x')
        plt.legend()

        plt.subplot(grid[0, 2])
        #plt.title("Linear Velocity {} m/s".format(self.v_history[-1]))
        plt.plot(self.v_history,c='tab:orange')
        plt.ylabel('v(t) [m/s]')
        plt.xlabel('sample')

        plt.subplot(grid[1, 2])
        #plt.title("Angular Velocity {} m/s".format(self.w_history[-1]))
        plt.plot(np.degrees(self.w_history),c='tab:orange')
        plt.ylabel('w(t) [deg/s]')
        plt.xlabel('sample')

        plt.tight_layout()

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

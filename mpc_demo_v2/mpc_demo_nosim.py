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
SIM_START_Y=0.5
SIM_START_H=0

from mpc_config import Params
P=Params()

# Classes
class MPC():

    def __init__(self):

        # State for the robot mathematical model [x,y,heading]
        self.state =  [SIM_START_X, SIM_START_Y, SIM_START_H]

        self.opt_u =  np.zeros((P.M,P.T))
        self.opt_u[0,:] = 1 #m/s
        self.opt_u[1,:] = np.radians(0) #rad/s

        # Interpolated Path to follow given waypoints
        #self.path = compute_path_from_wp([0,10,12,2,4,14],[0,0,2,10,12,12])
        self.path = compute_path_from_wp([0,3,4,6,10,13],
                                         [0,0,2,4,3,3],1)

        # Sim help vars
        self.sim_time=0
        self.x_history=[]
        self.y_history=[]
        self.h_history=[]
        self.v_history=[]
        self.w_history=[]
        self.predicted=None

        #Initialise plot
        plt.style.use("ggplot")
        self.fig = plt.figure()
        plt.ion()
        plt.show()

    def predict_motion(self):
        '''
        '''
        predicted=np.zeros(self.opt_u.shape)
        x=self.state[0]
        y=self.state[1]
        th=self.state[2]
        for idx,v,w in zip(range(len(self.opt_u[0,:])),self.opt_u[0,:],self.opt_u[1,:]):
            x = x+v*np.cos(th)*P.dt
            y = y+v*np.sin(th)*P.dt
            th= th +w*P.dt
            predicted[0,idx]=x
            predicted[1,idx]=y

        self.predicted = predicted

    def run(self):
        '''
        '''

        while 1:
            if self.state is not None:

                if np.sqrt((self.state[0]-self.path[0,-1])**2+(self.state[1]-self.path[1,-1])**2)<1:
                    print("Success! Goal Reached")
                    return

                #optimization loop
                start=time.time()
                self.opt_u = optimize(self.state,
                                        self.opt_u,
                                        self.path)

                # print("CVXPY Optimization Time: {:.4f}s".format(time.time()-start))

                self.update_sim(self.opt_u[0,1],self.opt_u[1,1])
                self.predict_motion()
                self.plot_sim()

    def update_sim(self,lin_v,ang_v):
        '''
        Updates state.

        :param lin_v: float
        :param ang_v: float
        '''

        self.state[0] = self.state[0] +lin_v*np.cos(self.state[2])*P.dt
        self.state[1] = self.state[1] +lin_v*np.sin(self.state[2])*P.dt
        self.state[2] = self.state[2] +ang_v*P.dt

    def plot_sim(self):

        self.sim_time = self.sim_time+P.dt
        self.x_history.append(self.state[0])
        self.y_history.append(self.state[1])
        self.h_history.append(self.state[2])
        self.v_history.append(self.opt_u[0,1])
        self.w_history.append(self.opt_u[1,1])

        plt.clf()

        grid = plt.GridSpec(2, 3)

        plt.subplot(grid[0:2, 0:2])
        plt.title("MPC Simulation \n" + "Simulation elapsed time {}s".format(self.sim_time))

        plt.plot(self.path[0,:],self.path[1,:], c='tab:orange',
                                                marker=".",
                                                label="reference track")

        plt.plot(self.x_history, self.y_history, c='tab:blue',
                                                 marker=".",
                                                 alpha=0.5,
                                                 label="vehicle trajectory")

        if self.predicted is not None:
            plt.plot(self.predicted[0,:], self.predicted[1,:], c='tab:green',
                                                     marker=".",
                                                     alpha=0.5,
                                                     label="mpc opt trajectory")

        # plt.plot(self.x_history[-1], self.y_history[-1], c='tab:blue',
        #                                                  marker=".",
        #                                                  markersize=12,
        #                                                  label="vehicle position")
        # plt.arrow(self.x_history[-1],
        #           self.y_history[-1],
        #           np.cos(self.h_history[-1]),
        #           np.sin(self.h_history[-1]),
        #           color='tab:blue',
        #           width=0.2,
        #           head_length=0.5,
        #           label="heading")

        plot_car(self.x_history[-1], self.y_history[-1], self.h_history[-1])

        plt.ylabel('map y')
        plt.yticks(np.arange(min(self.path[1,:])-1., max(self.path[1,:]+1.)+1, 1.0))
        plt.xlabel('map x')
        plt.xticks(np.arange(min(self.path[0,:])-1., max(self.path[0,:]+1.)+1, 1.0))
        plt.axis("equal")
        #plt.legend()

        plt.subplot(grid[0, 2])
        #plt.title("Linear Velocity {} m/s".format(self.v_history[-1]))
        plt.plot(self.v_history,c='tab:orange')
        locs, _ = plt.xticks()
        plt.xticks(locs[1:], locs[1:]*P.dt)
        plt.ylabel('v(t) [m/s]')
        plt.xlabel('t [s]')

        plt.subplot(grid[1, 2])
        #plt.title("Angular Velocity {} m/s".format(self.w_history[-1]))
        plt.plot(np.degrees(self.w_history),c='tab:orange')
        plt.ylabel('w(t) [deg/s]')
        locs, _ = plt.xticks()
        plt.xticks(locs[1:], locs[1:]*P.dt)
        plt.xlabel('t [s]')

        plt.tight_layout()

        plt.draw()
        plt.pause(0.1)


def plot_car(x, y, yaw):
    LENGTH = 1.0  # [m]
    WIDTH = 0.5  # [m]
    OFFSET = LENGTH/2  # [m]

    outline = np.array([[-OFFSET, (LENGTH - OFFSET), (LENGTH - OFFSET), -OFFSET, -OFFSET],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    Rotm = np.array([[np.cos(yaw), np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]])

    outline = (outline.T.dot(Rotm)).T

    outline[0, :] += x
    outline[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), 'tab:blue')



def do_sim():
    sim=MPC()
    try:
        sim.run()
    except Exception as e:
        sys.exit(e)

if __name__ == '__main__':
    do_sim()

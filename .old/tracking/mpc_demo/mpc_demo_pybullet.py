import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from utils import compute_path_from_wp
from cvxpy_mpc import optimize

import sys
import time

import pybullet as p
import time

def get_state(robotId):
    """
    """
    robPos, robOrn = p.getBasePositionAndOrientation(robotId)
    linVel,angVel = p.getBaseVelocity(robotId)

    return[robPos[0], robPos[1], p.getEulerFromQuaternion(robOrn)[2]]

def set_ctrl(robotId,v,w):
	"""
	"""
	L= 0.354
	R= 0.076/2

	rightWheelVelocity= (2*v+w*L)/(2*R)
	leftWheelVelocity = (2*v-w*L)/(2*R)

	p.setJointMotorControl2(robotId,0,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
	p.setJointMotorControl2(robotId,1,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)

def plot(path,x_history,y_history):
    """
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.title("MPC Tracking Results")

    plt.plot(path[0,:],path[1,:], c='tab:orange',marker=".",label="reference track")
    plt.plot(x_history, y_history, c='tab:blue',marker=".",alpha=0.5,label="vehicle trajectory")

    plt.legend()
    plt.show()

def run_sim():
    """
    """
    p.connect(p.GUI)

    start_offset = [0,2,0]
    start_orientation = p.getQuaternionFromEuler([0,0,0])
    turtle = p.loadURDF("turtlebot.urdf",start_offset, start_orientation)
    plane = p.loadURDF("plane.urdf")

    p.setRealTimeSimulation(1)
    p.setGravity(0,0,-10)

    # MPC time step
    dt = 0.25

    # starting guess output
    N = 5 #number of state variables
    M = 2 #number of control variables
    T = 20 #Prediction Horizon

    opt_u =  np.zeros((M,T))
    opt_u[0,:] = 1 #m/s
    opt_u[1,:] = np.radians(0) #rad/s

    # Interpolated Path to follow given waypoints
    path = compute_path_from_wp([0,10,12,2,4,14],[0,0,2,10,12,12])
    x_history=[]
    y_history=[]

    while (1):

        state =  get_state(turtle)
        x_history.append(state[0])
        y_history.append(state[1])

        #track path in bullet
        p.addUserDebugLine([state[0],state[1],0],[state[0],state[1],0.5],[1,0,0])

        if np.sqrt((state[0]-path[0,-1])**2+(state[1]-path[1,-1])**2)<1:
            print("Success! Goal Reached")
            set_ctrl(turtle,0,0)
            plot(path,x_history,y_history)
            p.disconnect()
            return

    	#optimization loop
        start=time.time()
        opt_u = optimize(state,opt_u,path)
        elapsed=time.time()-start
        print("CVXPY Optimization Time: {:.4f}s".format(elapsed))

        set_ctrl(turtle,opt_u[0,1],opt_u[1,1])

        if dt-elapsed>0:
            time.sleep(dt-elapsed)

if __name__ == '__main__':
    run_sim()

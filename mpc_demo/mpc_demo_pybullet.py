import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from utils import compute_path_from_wp
from cvxpy_mpc import optimize

from mpc_config import Params
P=Params()

import sys
import time

import pybullet as p
import time

def get_state(robotId):
    """
    """
    robPos, robOrn = p.getBasePositionAndOrientation(robotId)
    linVel,angVel = p.getBaseVelocity(robotId)

    return[robPos[0], robPos[1], linVel[0], p.getEulerFromQuaternion(robOrn)[2]]

def set_ctrl(robotId,currVel,acceleration,steeringAngle):

    gearRatio=1./21
    steering = [0,2]
    wheels = [8,15]
    maxForce = 50

    targetVelocity = currVel + acceleration*P.dt
    #targetVelocity=lastVel
    #print(targetVelocity)

    for wheel in wheels:
    	p.setJointMotorControl2(robotId,wheel,p.VELOCITY_CONTROL,targetVelocity=targetVelocity/gearRatio,force=maxForce)

    for steer in steering:
    	p.setJointMotorControl2(robotId,steer,p.POSITION_CONTROL,targetPosition=steeringAngle)

def plot_results(path,x_history,y_history):
    """
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.title("MPC Tracking Results")

    plt.plot(path[0,:],path[1,:], c='tab:orange',marker=".",label="reference track")
    plt.plot(x_history, y_history, c='tab:blue',marker=".",alpha=0.5,label="vehicle trajectory")
    plt.axis("equal")
    plt.legend()
    plt.show()

def run_sim():
    """
    """
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=-90, cameraPitch=-45, cameraTargetPosition=[-0.1,-0.0,0.65])

    p.resetSimulation()

    p.setGravity(0,0,-10)
    useRealTimeSim = 1

    p.setTimeStep(1./120.)
    p.setRealTimeSimulation(useRealTimeSim) # either this

    plane = p.loadURDF("racecar/plane.urdf")
    #track = p.loadSDF("racecar/f10_racecar/meshes/barca_track.sdf", globalScaling=1)

    car = p.loadURDF("racecar/f10_racecar/racecar_differential.urdf", [0,0,.3])
    for wheel in range(p.getNumJoints(car)):
    	print("joint[",wheel,"]=", p.getJointInfo(car,wheel))
    	p.setJointMotorControl2(car,wheel,p.VELOCITY_CONTROL,targetVelocity=0,force=0)
    	p.getJointInfo(car,wheel)

    c = p.createConstraint(car,9,car,11,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=1, maxForce=10000)

    c = p.createConstraint(car,10,car,13,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car,9,car,13,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car,16,car,18,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=1, maxForce=10000)


    c = p.createConstraint(car,16,car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car,17,car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=-1, maxForce=10000)

    c = p.createConstraint(car,1,car,18,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15, maxForce=10000)
    c = p.createConstraint(car,3,car,19,jointType=p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15,maxForce=10000)

    opt_u =  np.zeros((P.M,P.T))
    opt_u[0,:] = 1 #m/ss
    opt_u[1,:] = np.radians(0) #rad/

    # Interpolated Path to follow given waypoints
    path = compute_path_from_wp([0,3,4,6,10,11,12,6,1,0],
                                [0,0,2,4,3,3,-1,-6,-2,-2],0.5)

    for x_,y_ in zip(path[0,:],path[1,:]):
        p.addUserDebugLine([x_,y_,0],[x_,y_,0.33],[0,0,1])

    x_history=[]
    y_history=[]

    time.sleep(0.5)
    input("Press Enter to continue...")
    while (1):

        state =  get_state(car)
        x_history.append(state[0])
        y_history.append(state[1])

        #add 1 timestep delay to input
        state[0]=state[0]+state[2]*np.cos(state[3])*P.dt
        state[1]=state[1]+state[2]*np.sin(state[3])*P.dt
        state[2]=state[2]+opt_u[0,0]*P.dt
        state[3]=state[3]+opt_u[0,0]*np.tan(opt_u[1,0])/0.3*P.dt

        #track path in bullet
        p.addUserDebugLine([state[0],state[1],0],[state[0],state[1],0.5],[1,0,0])

        if np.sqrt((state[0]-path[0,-1])**2+(state[1]-path[1,-1])**2)<1:
            print("Success! Goal Reached")
            set_ctrl(car,0,0,0)
            plot_results(path,x_history,y_history)
            p.disconnect()
            return

    	#optimization loop
        start=time.time()
        opt_u = optimize(state,opt_u,path,ref_vel=1.0)
        elapsed=time.time()-start
        print("CVXPY Optimization Time: {:.4f}s".format(elapsed))

        set_ctrl(car,state[2],opt_u[0,1],opt_u[1,1])

        if P.dt-elapsed>0:
            time.sleep(P.dt-elapsed)

if __name__ == '__main__':
    run_sim()

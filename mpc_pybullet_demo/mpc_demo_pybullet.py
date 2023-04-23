import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from mpcpy.utils import compute_path_from_wp
import mpcpy

P = mpcpy.Params()

import sys
import time

import pybullet as p
import time


def get_state(robotId):
    """ """
    robPos, robOrn = p.getBasePositionAndOrientation(robotId)
    linVel, angVel = p.getBaseVelocity(robotId)

    return np.array(
        [
            robPos[0],
            robPos[1],
            np.sqrt(linVel[0] ** 2 + linVel[1] ** 2),
            p.getEulerFromQuaternion(robOrn)[2],
        ]
    )


def set_ctrl(robotId, currVel, acceleration, steeringAngle):
    gearRatio = 1.0 / 21
    steering = [0, 2]
    wheels = [8, 15]
    maxForce = 50

    targetVelocity = currVel + acceleration * P.DT

    for wheel in wheels:
        p.setJointMotorControl2(
            robotId,
            wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=targetVelocity / gearRatio,
            force=maxForce,
        )

    for steer in steering:
        p.setJointMotorControl2(
            robotId, steer, p.POSITION_CONTROL, targetPosition=steeringAngle
        )


def plot_results(path, x_history, y_history):
    """ """
    plt.style.use("ggplot")
    plt.figure()
    plt.title("MPC Tracking Results")

    plt.plot(
        path[0, :], path[1, :], c="tab:orange", marker=".", label="reference track"
    )
    plt.plot(
        x_history,
        y_history,
        c="tab:blue",
        marker=".",
        alpha=0.5,
        label="vehicle trajectory",
    )
    plt.axis("equal")
    plt.legend()
    plt.show()


def run_sim():
    """ """
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=-90,
        cameraPitch=-45,
        cameraTargetPosition=[-0.1, -0.0, 0.65],
    )

    p.resetSimulation()

    p.setGravity(0, 0, -10)
    useRealTimeSim = 1

    p.setTimeStep(1.0 / 120.0)
    p.setRealTimeSimulation(useRealTimeSim)  # either this

    plane = p.loadURDF("racecar/plane.urdf")
    # track = p.loadSDF("racecar/f10_racecar/meshes/barca_track.sdf", globalScaling=1)

    car = p.loadURDF("racecar/f10_racecar/racecar_differential.urdf", [0, 0.3, 0.3])
    for wheel in range(p.getNumJoints(car)):
        # print("joint[",wheel,"]=", p.getJointInfo(car,wheel))
        p.setJointMotorControl2(
            car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0
        )
        p.getJointInfo(car, wheel)

    c = p.createConstraint(
        car,
        9,
        car,
        11,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(
        car,
        10,
        car,
        13,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(
        car,
        9,
        car,
        13,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(
        car,
        16,
        car,
        18,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=1, maxForce=10000)

    c = p.createConstraint(
        car,
        16,
        car,
        19,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(
        car,
        17,
        car,
        19,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, maxForce=10000)

    c = p.createConstraint(
        car,
        1,
        car,
        18,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
    c = p.createConstraint(
        car,
        3,
        car,
        19,
        jointType=p.JOINT_GEAR,
        jointAxis=[0, 1, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    # Interpolated Path to follow given waypoints
    path = compute_path_from_wp(
        [0, 3, 4, 6, 10, 11, 12, 6, 1, 0],
        [0, 0, 2, 4, 3, 3, -1, -6, -2, -2],
        P.path_tick,
    )

    for x_, y_ in zip(path[0, :], path[1, :]):
        p.addUserDebugLine([x_, y_, 0], [x_, y_, 0.33], [0, 0, 1])

    # starting guess
    action = np.zeros(P.M)
    action[0] = P.MAX_ACC / 2  # a
    action[1] = 0.0  # delta

    # Cost Matrices
    Q = [20, 20, 10, 20]  # state error cost [x,y,v,yaw]
    Qf = [30, 30, 30, 30]  # state error cost at final timestep [x,y,v,yaw]
    R = [10, 10]  # input cost [acc ,steer]
    P = [10, 10]  # input rate of change cost [acc ,steer]

    mpc = mpcpy.MPC(Q, Qf, R, P)
    x_history = []
    y_history = []

    input("\033[92m Press Enter to continue... \033[0m")

    while 1:
        state = get_state(car)
        x_history.append(state[0])
        y_history.append(state[1])

        # track path in bullet
        p.addUserDebugLine(
            [state[0], state[1], 0], [state[0], state[1], 0.5], [1, 0, 0]
        )

        if np.sqrt((state[0] - path[0, -1]) ** 2 + (state[1] - path[1, -1]) ** 2) < 0.2:
            print("Success! Goal Reached")
            set_ctrl(car, 0, 0, 0)
            plot_results(path, x_history, y_history)
            input("Press Enter to continue...")
            p.disconnect()
            return

        # for MPC base link frame is used:
        # so x, y, yaw are 0.0, but speed is the same
        ego_state = [0.0, 0.0, state[2], 0.0]

        # to account for MPC latency
        # we simulate one timestep delay
        ego_state[0] = ego_state[0] + ego_state[2] * np.cos(ego_state[3]) * P.DT
        ego_state[1] = ego_state[1] + ego_state[2] * np.sin(ego_state[3]) * P.DT
        ego_state[2] = ego_state[2] + action[0] * P.DT
        ego_state[3] = ego_state[3] + action[0] * np.tan(action[1]) / P.L * P.DT

        # optimization loop
        start = time.time()

        # State Matrices
        A, B, C = mpcpy.get_linear_model_matrices(ego_state, action)

        # Get Reference_traj
        # NOTE: inputs are in world frame
        target, _ = mpcpy.get_ref_trajectory(ego_state, path, P.TARGET_SPEED)

        # MPC step
        action = mpc.optimize_linearized_model(
            A, B, C, state, target, time_horizon=P.T, verbose=False
        )

        elapsed = time.time() - start
        print("CVXPY Optimization Time: {:.4f}s".format(elapsed))

        set_ctrl(car, state[2], acc, steer)

        if P.DT - elapsed > 0:
            time.sleep(P.DT - elapsed)


if __name__ == "__main__":
    run_sim()

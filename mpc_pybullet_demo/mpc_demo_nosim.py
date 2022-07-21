#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint

from mpcpy.utils import compute_path_from_wp
import mpcpy

import sys
import time

# Robot Starting position
SIM_START_X = 0.0
SIM_START_Y = 0.5
SIM_START_V = 0.0
SIM_START_H = 0.0
L = 0.3

P = mpcpy.Params()

# Params
VEL = 1.0  # m/s

# Classes
class MPCSim:
    def __init__(self):

        # State for the robot mathematical model [x,y,heading]
        self.state = np.array([SIM_START_X, SIM_START_Y, SIM_START_V, SIM_START_H])

        # starting guess
        self.action = np.zeros(P.M)
        self.action[0] = P.MAX_ACC / 2  # a
        self.action[1] = 0.0  # delta

        self.opt_u = np.zeros((P.M, P.T))

        # Cost Matrices
        Q = np.diag([20, 20, 10, 20])  # state error cost
        Qf = np.diag([30, 30, 30, 30])  # state final error cost
        R = np.diag([10, 10])  # input cost
        R_ = np.diag([10, 10])  # input rate of change cost

        self.mpc = mpcpy.MPC(P.N, P.M, Q, R)

        # Interpolated Path to follow given waypoints
        self.path = compute_path_from_wp(
            [0, 3, 4, 6, 10, 12, 13, 13, 6, 1, 0],
            [0, 0, 2, 4, 3, 3, -1, -2, -6, -2, -2],
            P.path_tick,
        )

        # Sim help vars
        self.sim_time = 0
        self.x_history = []
        self.y_history = []
        self.v_history = []
        self.h_history = []
        self.a_history = []
        self.d_history = []
        self.predicted = None

        # Initialise plot
        plt.style.use("ggplot")
        self.fig = plt.figure()
        plt.ion()
        plt.show()

    def preview(self, mpc_out):
        """
        [TODO:summary]

        [TODO:description]
        """
        predicted = np.zeros(self.opt_u.shape)
        predicted[:, :] = mpc_out[0:2, 1:]
        Rotm = np.array(
            [
                [np.cos(self.state[3]), np.sin(self.state[3])],
                [-np.sin(self.state[3]), np.cos(self.state[3])],
            ]
        )
        predicted = (predicted.T.dot(Rotm)).T
        predicted[0, :] += self.state[0]
        predicted[1, :] += self.state[1]
        self.predicted = predicted

    def run(self):
        """
        [TODO:summary]

        [TODO:description]
        """
        self.plot_sim()
        input("Press Enter to continue...")
        while 1:
            if (
                np.sqrt(
                    (self.state[0] - self.path[0, -1]) ** 2
                    + (self.state[1] - self.path[1, -1]) ** 2
                )
                < 0.5
            ):
                print("Success! Goal Reached")
                input("Press Enter to continue...")
                return
            # optimization loop
            # start=time.time()
            # dynamycs w.r.t robot frame
            curr_state = np.array([0, 0, self.state[2], 0])
            # State Matrices
            A, B, C = mpcpy.get_linear_model_matrices(curr_state, self.action)
            # Get Reference_traj -> inputs are in worldframe
            target, _ = mpcpy.get_ref_trajectory(
                self.state, self.path, VEL, dl=P.path_tick
            )

            x_mpc, u_mpc = self.mpc.optimize_linearized_model(
                A,
                B,
                C,
                curr_state,
                target,
                time_horizon=P.T,
                verbose=False,
            )
            self.opt_u = np.vstack(
                (
                    np.array(u_mpc.value[0, :]).flatten(),
                    (np.array(u_mpc.value[1, :]).flatten()),
                )
            )
            self.action[:] = [u_mpc.value[0, 0], u_mpc.value[1, 0]]
            # print("CVXPY Optimization Time: {:.4f}s".format(time.time()-start))
            self.predict([self.action[0], self.action[1]])
            self.preview(x_mpc.value)
            self.plot_sim()

    def predict(self, u):
        def kinematics_model(x, t, u):
            dxdt = x[2] * np.cos(x[3])
            dydt = x[2] * np.sin(x[3])
            dvdt = u[0]
            dtheta0dt = x[2] * np.tan(u[1]) / P.L
            dqdt = [dxdt, dydt, dvdt, dtheta0dt]
            return dqdt

        # solve ODE
        tspan = [0, P.DT]
        self.state = odeint(kinematics_model, self.state, tspan, args=(u[:],))[1]

    def plot_sim(self):
        """
        [TODO:summary]

        [TODO:description]
        """
        self.sim_time = self.sim_time + P.DT
        self.x_history.append(self.state[0])
        self.y_history.append(self.state[1])
        self.v_history.append(self.state[2])
        self.h_history.append(self.state[3])
        self.a_history.append(self.opt_u[0, 1])
        self.d_history.append(self.opt_u[1, 1])

        plt.clf()

        grid = plt.GridSpec(2, 3)

        plt.subplot(grid[0:2, 0:2])
        plt.title(
            "MPC Simulation \n" + "Simulation elapsed time {}s".format(self.sim_time)
        )

        plt.plot(
            self.path[0, :],
            self.path[1, :],
            c="tab:orange",
            marker=".",
            label="reference track",
        )

        plt.plot(
            self.x_history,
            self.y_history,
            c="tab:blue",
            marker=".",
            alpha=0.5,
            label="vehicle trajectory",
        )

        if self.predicted is not None:
            plt.plot(
                self.predicted[0, :],
                self.predicted[1, :],
                c="tab:green",
                marker="+",
                alpha=0.5,
                label="mpc opt trajectory",
            )

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

        plt.ylabel("map y")
        plt.yticks(
            np.arange(min(self.path[1, :]) - 1.0, max(self.path[1, :] + 1.0) + 1, 1.0)
        )
        plt.xlabel("map x")
        plt.xticks(
            np.arange(min(self.path[0, :]) - 1.0, max(self.path[0, :] + 1.0) + 1, 1.0)
        )
        plt.axis("equal")
        # plt.legend()

        plt.subplot(grid[0, 2])
        # plt.title("Linear Velocity {} m/s".format(self.v_history[-1]))
        plt.plot(self.a_history, c="tab:orange")
        locs, _ = plt.xticks()
        plt.xticks(locs[1:], locs[1:] * P.DT)
        plt.ylabel("a(t) [m/ss]")
        plt.xlabel("t [s]")

        plt.subplot(grid[1, 2])
        # plt.title("Angular Velocity {} m/s".format(self.w_history[-1]))
        plt.plot(np.degrees(self.d_history), c="tab:orange")
        plt.ylabel("gamma(t) [deg]")
        locs, _ = plt.xticks()
        plt.xticks(locs[1:], locs[1:] * P.DT)
        plt.xlabel("t [s]")

        plt.tight_layout()

        plt.draw()
        plt.pause(0.1)


def plot_car(x, y, yaw):
    """
    [TODO:summary]

    [TODO:description]

    Parameters
    ----------
    x : [TODO:type]
        [TODO:description]
    y : [TODO:type]
        [TODO:description]
    yaw : [TODO:type]
        [TODO:description]
    """
    LENGTH = 0.5  # [m]
    WIDTH = 0.25  # [m]
    OFFSET = LENGTH  # [m]

    outline = np.array(
        [
            [-OFFSET, (LENGTH - OFFSET), (LENGTH - OFFSET), -OFFSET, -OFFSET],
            [WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
        ]
    )

    Rotm = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])

    outline = (outline.T.dot(Rotm)).T

    outline[0, :] += x
    outline[1, :] += y

    plt.plot(
        np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), "tab:blue"
    )


def do_sim():
    sim = MPCSim()
    try:
        sim.run()
    except Exception as e:
        sys.exit(e)


if __name__ == "__main__":
    do_sim()

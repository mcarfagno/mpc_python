#! /usr/bin/env python

import sys

import matplotlib.pyplot as plt
import numpy as np
from cvxpy_mpc import MPC, VehicleModel
from cvxpy_mpc.utils import compute_path_from_wp, get_ref_trajectory
from scipy.integrate import odeint

# Robot Starting position
SIM_START_X = 0.0
SIM_START_Y = 0.5
SIM_START_V = 0.0
SIM_START_H = 0.0


# Params
TARGET_VEL = 1.0  # m/s
T = 5  # Prediction Horizon [s]
DT = 0.2  # discretization step [s]
L = 0.3  # vehicle wheelbase [m]


# Classes
class MPCSim:
    def __init__(self):
        # State of the robot [x,y,v, heading]
        self.state = np.array([SIM_START_X, SIM_START_Y, SIM_START_V, SIM_START_H])

        # helper variable to keep track of mpc output
        # starting condition is 0,0
        self.control = np.zeros(2)

        self.K = int(T / DT)

        Q = [20, 20, 10, 20]  # state error cost
        Qf = [30, 30, 30, 30]  # state final error cost
        R = [10, 10]  # input cost
        P = [10, 10]  # input rate of change cost
        self.mpc = MPC(VehicleModel(), T, DT, Q, Qf, R, P)

        # Path from waypoint interpolation
        self.path = compute_path_from_wp(
            [0, 3, 4, 6, 10, 12, 13, 13, 6, 1, 0],
            [0, 0, 2, 4, 3, 3, -1, -2, -6, -2, -2],
            0.05,
        )

        # Helper variables to keep track of the sim
        self.sim_time = 0
        self.x_history = []
        self.y_history = []
        self.v_history = []
        self.h_history = []
        self.a_history = []
        self.d_history = []
        self.optimized_trajectory = None

        # Initialise plot
        plt.style.use("ggplot")
        self.fig = plt.figure()
        plt.ion()
        plt.show()

    def ego_to_global(self, mpc_out):
        """
        transforms optimized trajectory XY points from ego(car) reference
        into global(map) frame

        Args:
            mpc_out ():
        """
        trajectory = np.zeros((2, self.K))
        trajectory[:, :] = mpc_out[0:2, 1:]
        Rotm = np.array(
            [
                [np.cos(self.state[3]), np.sin(self.state[3])],
                [-np.sin(self.state[3]), np.cos(self.state[3])],
            ]
        )
        trajectory = (trajectory.T.dot(Rotm)).T
        trajectory[0, :] += self.state[0]
        trajectory[1, :] += self.state[1]
        return trajectory

    def run(self):
        """
        [TODO:summary]

        [TODO:description]
        """
        self.plot_sim()
        input("Press Enter to continue...")
        try:
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

                # Get Reference_traj -> inputs are in worldframe
                target = get_ref_trajectory(self.state, self.path, TARGET_VEL, T, DT)

                # dynamycs w.r.t robot frame
                curr_state = np.array([0, 0, self.state[2], 0])
                x_mpc, u_mpc = self.mpc.step(
                    curr_state,
                    target,
                    self.control,
                    verbose=False,
                )
                # print("CVXPY Optimization Time: {:.4f}s".format(time.time()-start))
                # only the first one is used to advance the simulation

                self.control[:] = [u_mpc[0, 0], u_mpc[1, 0]]
                self.state = self.predict_next_state(
                    self.state, [self.control[0], self.control[1]], DT
                )

                # use the optimizer output to preview the predicted state trajectory
                self.optimized_trajectory = self.ego_to_global(x_mpc)
                self.plot_sim()
        except KeyboardInterrupt:
            pass

    def predict_next_state(self, state, u, dt):
        def kinematics_model(x, t, u):
            dxdt = x[2] * np.cos(x[3])
            dydt = x[2] * np.sin(x[3])
            dvdt = u[0]
            dthetadt = x[2] * np.tan(u[1]) / L
            dqdt = [dxdt, dydt, dvdt, dthetadt]
            return dqdt

        # solve ODE
        tspan = [0, dt]
        new_state = odeint(kinematics_model, state, tspan, args=(u[:],))[1]
        return new_state

    def plot_sim(self):
        self.sim_time = self.sim_time + DT
        self.x_history.append(self.state[0])
        self.y_history.append(self.state[1])
        self.v_history.append(self.state[2])
        self.h_history.append(self.state[3])
        self.a_history.append(self.control[0])
        self.d_history.append(self.control[1])

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

        if self.optimized_trajectory is not None:
            plt.plot(
                self.optimized_trajectory[0, :],
                self.optimized_trajectory[1, :],
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
        plt.xticks(locs[1:], locs[1:] * DT)
        plt.ylabel("a(t) [m/ss]")
        plt.xlabel("t [s]")

        plt.subplot(grid[1, 2])
        # plt.title("Angular Velocity {} m/s".format(self.w_history[-1]))
        plt.plot(np.degrees(self.d_history), c="tab:orange")
        plt.ylabel("gamma(t) [deg]")
        locs, _ = plt.xticks()
        plt.xticks(locs[1:], locs[1:] * DT)
        plt.xlabel("t [s]")

        plt.tight_layout()

        plt.draw()
        plt.pause(0.1)


def plot_car(x, y, yaw):
    """

    Args:
        x ():
        y ():
        yaw ():
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

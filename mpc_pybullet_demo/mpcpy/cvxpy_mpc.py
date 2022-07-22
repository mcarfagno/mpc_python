import numpy as np

np.seterr(divide="ignore", invalid="ignore")

from scipy.integrate import odeint
from scipy.interpolate import interp1d
import cvxpy as opt

from .utils import *

from .mpc_config import Params

P = Params()


def get_linear_model_matrices(x_bar, u_bar):
    """
    Computes the LTI approximated state space model x' = Ax + Bu + C
    """

    x = x_bar[0]
    y = x_bar[1]
    v = x_bar[2]
    theta = x_bar[3]

    a = u_bar[0]
    delta = u_bar[1]

    ct = np.cos(theta)
    st = np.sin(theta)
    cd = np.cos(delta)
    td = np.tan(delta)

    A = np.zeros((P.N, P.N))
    A[0, 2] = ct
    A[0, 3] = -v * st
    A[1, 2] = st
    A[1, 3] = v * ct
    A[3, 2] = v * td / P.L
    A_lin = np.eye(P.N) + P.DT * A

    B = np.zeros((P.N, P.M))
    B[2, 0] = 1
    B[3, 1] = v / (P.L * cd**2)
    B_lin = P.DT * B

    f_xu = np.array([v * ct, v * st, a, v * td / P.L]).reshape(P.N, 1)
    C_lin = (
        P.DT
        * (
            f_xu - np.dot(A, x_bar.reshape(P.N, 1)) - np.dot(B, u_bar.reshape(P.M, 1))
        ).flatten()
    )

    # return np.round(A_lin,6), np.round(B_lin,6), np.round(C_lin,6)
    return A_lin, B_lin, C_lin


class MPC:
    def __init__(self, N, M, Q, R):
        """ """
        self.state_len = N
        self.action_len = M
        self.state_cost = Q
        self.action_cost = R

    def optimize_linearized_model(
        self,
        A,
        B,
        C,
        initial_state,
        target,
        time_horizon=10,
        Q=None,
        R=None,
        verbose=False,
    ):
        """
        Optimisation problem defined for the linearised model,
        :param A:
        :param B:
        :param C:
        :param initial_state:
        :param Q:
        :param R:
        :param target:
        :param time_horizon:
        :param verbose:
        :return:
        """

        assert len(initial_state) == self.state_len

        if Q == None or R == None:
            Q = self.state_cost
            R = self.action_cost

        # Create variables
        x = opt.Variable((self.state_len, time_horizon + 1), name="states")
        u = opt.Variable((self.action_len, time_horizon), name="actions")

        # Loop through the entire time_horizon and append costs
        cost_function = []

        for t in range(time_horizon):

            _cost = opt.quad_form(target[:, t + 1] - x[:, t + 1], Q) + opt.quad_form(
                u[:, t], R
            )

            _constraints = [
                x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C,
                u[0, t] >= -P.MAX_ACC,
                u[0, t] <= P.MAX_ACC,
                u[1, t] >= -P.MAX_STEER,
                u[1, t] <= P.MAX_STEER,
            ]
            # opt.norm(target[:, t + 1] - x[:, t + 1], 1) <= 0.1]

            # Actuation rate of change
            if t < (time_horizon - 1):
                _cost += opt.quad_form(u[:, t + 1] - u[:, t], R * 1)
                _constraints += [opt.abs(u[0, t + 1] - u[0, t]) / P.DT <= P.MAX_D_ACC]
                _constraints += [opt.abs(u[1, t + 1] - u[1, t]) / P.DT <= P.MAX_D_STEER]

            if t == 0:
                # _constraints += [opt.norm(target[:, time_horizon] - x[:, time_horizon], 1) <= 0.01,
                #                x[:, 0] == initial_state]
                _constraints += [x[:, 0] == initial_state]

            cost_function.append(
                opt.Problem(opt.Minimize(_cost), constraints=_constraints)
            )

        # Add final cost
        problem = sum(cost_function)

        # Minimize Problem
        problem.solve(verbose=verbose, solver=opt.OSQP)
        return x, u

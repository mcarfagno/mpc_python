import numpy as np

np.seterr(divide="ignore", invalid="ignore")

import cvxpy as opt


class MPC:
    def __init__(
        self, vehicle, T, DT, state_cost, final_state_cost, input_cost, input_rate_cost
    ):
        """
        :param vehicle:
        :param T:
        :param DT:
        :param state_cost:
        :param final_state_cost:
        :param input_cost:
        :param input_rate_cost:
        """

        self.nx = 4  # number of state vars
        self.nu = 2  # umber of input/control vars

        if len(state_cost) != self.nx:
            raise ValueError(f"State Error cost matrix shuld be of size {self.nx}")
        if len(final_state_cost) != self.nx:
            raise ValueError(f"End State Error cost matrix shuld be of size {self.nx}")
        if len(input_cost) != self.nu:
            raise ValueError(f"Control Effort cost matrix shuld be of size {self.nu}")
        if len(input_rate_cost) != self.nu:
            raise ValueError(
                f"Control Effort Difference cost matrix shuld be of size {self.nu}"
            )

        self.vehicle = vehicle
        self.dt = DT
        self.control_horizon = int(T / DT)
        self.Q = np.diag(state_cost)
        self.Qf = np.diag(final_state_cost)
        self.R = np.diag(input_cost)
        self.P = np.diag(input_rate_cost)

    def get_linear_model_matrices(self, x_bar, u_bar):
        """
        Computes the LTI approximated state space model x' = Ax + Bu + C
        :param x_bar:
        :param u_bar:
        :return:
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

        A = np.zeros((self.nx, self.nx))
        A[0, 2] = ct
        A[0, 3] = -v * st
        A[1, 2] = st
        A[1, 3] = v * ct
        A[3, 2] = v * td / self.vehicle.wheelbase
        A_lin = np.eye(self.nx) + self.dt * A

        B = np.zeros((self.nx, self.nu))
        B[2, 0] = 1
        B[3, 1] = v / (self.vehicle.wheelbase * cd**2)
        B_lin = self.dt * B

        f_xu = np.array([v * ct, v * st, a, v * td / self.vehicle.wheelbase]).reshape(
            self.nx, 1
        )
        C_lin = (
            self.dt
            * (
                f_xu
                - np.dot(A, x_bar.reshape(self.nx, 1))
                - np.dot(B, u_bar.reshape(self.nu, 1))
            ).flatten()
        )
        return A_lin, B_lin, C_lin

    def step(
        self,
        initial_state,
        target,
        prev_cmd,
        verbose=False,
    ):
        """
        Optimisation problem defined for the linearised model,
        :param initial_state:
        :param target:
        :param verbose:
        :return:
        """

        assert len(initial_state) == self.nx

        # Create variables
        x = opt.Variable((self.nx, self.control_horizon + 1), name="states")
        u = opt.Variable((self.nu, self.control_horizon), name="actions")
        cost = 0
        constr = []

        A, B, C = self.get_linear_model_matrices(initial_state, prev_cmd)

        for k in range(self.control_horizon):
            cost += opt.quad_form(target[:, k] - x[:, k + 1], self.Q)
            cost += opt.quad_form(u[:, k], self.R)

            # Actuation rate of change
            if k < (self.control_horizon - 1):
                cost += opt.quad_form(u[:, k + 1] - u[:, k], self.P)

            # Kinematics Constrains
            constr += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + C]

            # Actuation rate of change bounds
            if k < (self.control_horizon - 1):
                constr += [
                    opt.abs(u[0, k + 1] - u[0, k]) / self.dt <= self.vehicle.max_d_acc
                ]
                constr += [
                    opt.abs(u[1, k + 1] - u[1, k]) / self.dt <= self.vehicle.max_d_steer
                ]

        # Final Point tracking
        cost += opt.quad_form(x[:, -1] - target[:, -1], self.Qf)

        # initial state
        constr += [x[:, 0] == initial_state]

        # actuation bounds
        constr += [opt.abs(u[:, 0]) <= self.vehicle.max_acc]
        constr += [opt.abs(u[:, 1]) <= self.vehicle.max_steer]

        prob = opt.Problem(opt.Minimize(cost), constr)
        solution = prob.solve(solver=opt.OSQP, warm_start=True, verbose=False)
        return x, u

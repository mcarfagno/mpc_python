from __future__ import annotations

import cvxpy as opt
import numpy as np
import numpy.typing as npt

from .vehicle_model import VehicleModel


class MPC:
    def __init__(
        self,
        vehicle: VehicleModel,
        T: float,
        DT: float,
        state_cost: list[float],
        final_state_cost: list[float],
        input_cost: list[float],
        input_rate_cost: list[float],
    ) -> None:
        self.nx: int = 5  # [X, Y, V, Theta, Steer_Angle]
        self.nu: int = 2  # [Accel, Steer_Rate]

        if len(state_cost) != self.nx:
            raise ValueError(f"State Error cost matrix should be of size {self.nx}")
        if len(final_state_cost) != self.nx:
            raise ValueError(f"End State Error cost matrix should be of size {self.nx}")
        if len(input_cost) != self.nu:
            raise ValueError(f"Control Effort cost matrix should be of size {self.nu}")
        if len(input_rate_cost) != self.nu:
            raise ValueError(
                f"Control Effort Difference cost matrix should be of size {self.nu}"
            )

        self.vehicle: VehicleModel = vehicle
        self.dt: float = DT
        self.control_horizon: int = int(T / DT)

        self.Q: npt.NDArray[np.float64] = np.diag(state_cost)
        self.Qf: npt.NDArray[np.float64] = np.diag(final_state_cost)
        self.R: npt.NDArray[np.float64] = np.diag(input_cost)
        self.Rr: npt.NDArray[np.float64] = np.diag(input_rate_cost)

        # CVXPY vars
        self.x: opt.Variable = opt.Variable(
            (self.nx, self.control_horizon + 1), name="states"
        )
        self.u: opt.Variable = opt.Variable(
            (self.nu, self.control_horizon), name="actions"
        )

        # CVXPY params (placeholder for run-time data)
        # We use params because raw values force CVXPY to re-parse the problem in Python every loop (slow).
        # Using dedicated Parameters allows CVXPY to lock down the matrix structure at startup.
        # At runtime, we only update the parameter '.value' (fast).
        self.initial_state_param: opt.Parameter = opt.Parameter(self.nx, name="x0")
        self.last_cmd_param: opt.Parameter = opt.Parameter(
            self.nu, name="last_applied_command"
        )

        self.A_params: list[opt.Parameter] = [
            opt.Parameter((self.nx, self.nx), name=f"A_{k}")
            for k in range(self.control_horizon)
        ]
        self.B_params: list[opt.Parameter] = [
            opt.Parameter((self.nx, self.nu), name=f"B_{k}")
            for k in range(self.control_horizon)
        ]
        self.C_params: list[opt.Parameter] = [
            opt.Parameter(self.nx, name=f"C_{k}") for k in range(self.control_horizon)
        ]

        # TARGET params
        # done this way to help make the cross-track error Disciplined Parametrized Programming (DPP) compliant...
        # see https://www.cvxpy.org/tutorial/dpp/index.html
        self.cos_param = opt.Parameter(self.control_horizon + 1)
        self.sin_param = opt.Parameter(self.control_horizon + 1)
        self.p_along_ref_param = opt.Parameter(self.control_horizon + 1)
        self.p_cross_ref_param = opt.Parameter(self.control_horizon + 1)
        self.v_ref_param = opt.Parameter(self.control_horizon + 1)
        self.theta_ref_param = opt.Parameter(self.control_horizon + 1)
        self.delta_ref_param = opt.Parameter(self.control_horizon + 1)

        # optimised vars
        self.prev_cmd: npt.NDArray[np.float64] | None = None
        self.prev_trajectory: npt.NDArray[np.float64] | None = None

        # build the problem ONCE
        self.prob: opt.Problem = self.make_mpc_problem()

    def compute_linear_model_matrices(
        self, x_bar: npt.NDArray[np.float64], u_bar: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:

        v = x_bar[2]
        theta = x_bar[3]
        delta = x_bar[4]

        a = u_bar[0]
        steer_rate = u_bar[1]

        # compute trigonometric functions once
        ct = np.cos(theta)
        st = np.sin(theta)
        cd = np.cos(delta)
        td = np.tan(delta)
        L = self.vehicle.wheelbase

        A = np.zeros((self.nx, self.nx))
        A[0, 2] = ct
        A[0, 3] = -v * st

        A[1, 2] = st
        A[1, 3] = v * ct

        A[3, 2] = td / L
        A[3, 4] = v / (L * cd**2)
        A_lin = np.eye(self.nx) + self.dt * A

        B = np.zeros((self.nx, self.nu))
        B[2, 0] = 1
        B[4, 1] = 1
        B_lin = self.dt * B

        f_xu = np.array([v * ct, v * st, a, v * td / L, steer_rate]).reshape(self.nx, 1)
        C_lin = (
            self.dt
            * (
                f_xu
                - np.dot(A, x_bar.reshape(self.nx, 1))
                - np.dot(B, u_bar.reshape(self.nu, 1))
            ).flatten()
        )
        return A_lin, B_lin, C_lin

    def make_mpc_problem(self) -> opt.Problem:

        cost = 0
        constr = []

        # Tracking error cost
        for k in range(self.control_horizon):

            # Kinematics constrains
            # Note each step uses the LTV matrix for that step
            constr += [
                self.x[:, k + 1]
                == self.A_params[k] @ self.x[:, k]
                + self.B_params[k] @ self.u[:, k]
                + self.C_params[k]
            ]

            # XY tracking does NOT make much sense in autonomous driving...
            # Instead we care how much off to the side we are w.r.t the track
            #
            # The standard cross-track and along-track errors are calculated by projecting position errors onto the track point
            # $\theta_{\text{ref}}$:$$e_{\text{along}} = \cos(\theta_{\text{ref}})(x - x_{\text{ref}}) + \sin(\theta_{\text{ref}})(y - y_{\text{ref}})
            # $e_{\text{cross}} = -\sin(\theta_{\text{ref}})(x - x_{\text{ref}}) + \cos(\theta_{\text{ref}})(y - y_{\text{ref}})$
            # we expand that and get the following Algebraic problem:

            # Algebraic along-track and cross-track expressions
            # We will fill the values when the reference is provided, that is why they are params
            e_along = (
                self.cos_param[k] * self.x[0, k]
                + self.sin_param[k] * self.x[1, k]
                - self.p_along_ref_param[k]
            )
            e_cross = (
                -self.sin_param[k] * self.x[0, k]
                + self.cos_param[k] * self.x[1, k]
                - self.p_cross_ref_param[k]
            )

            # error vector
            e = opt.vstack(
                [
                    e_along,
                    e_cross,
                    self.x[2, k] - self.v_ref_param[k],
                    self.x[3, k] - self.theta_ref_param[k],
                    self.x[4, k] - self.delta_ref_param[k],
                ]
            )
            cost += opt.quad_form(e, self.Q)

            # Actuation magnitude cost
            cost += opt.quad_form(self.u[:, k], self.R)

            # Actuation rate cost
            if k == 0:
                cost += opt.quad_form(self.u[:, 0] - self.last_cmd_param, self.Rr)
            else:
                cost += opt.quad_form(self.u[:, k] - self.u[:, k - 1], self.Rr)

        # Final point tracking cost
        e_along_f = (
            self.cos_param[-1] * self.x[0, -1]
            + self.sin_param[-1] * self.x[1, -1]
            - self.p_along_ref_param[-1]
        )
        e_cross_f = (
            -self.sin_param[-1] * self.x[0, -1]
            + self.cos_param[-1] * self.x[1, -1]
            - self.p_cross_ref_param[-1]
        )

        e_f = opt.vstack(
            [
                e_along_f,
                e_cross_f,
                self.x[2, -1] - self.v_ref_param[-1],
                self.x[3, -1] - self.theta_ref_param[-1],
                self.x[4, -1] - self.delta_ref_param[-1],
            ]
        )
        cost += opt.quad_form(e_f, self.Qf)

        # Initial state
        constr += [self.x[:, 0] == self.initial_state_param]

        constr += [opt.abs(self.x[2, :]) <= self.vehicle.max_speed]
        constr += [opt.abs(self.x[4, :]) <= self.vehicle.max_steer]

        constr += [opt.abs(self.u[0, :]) <= self.vehicle.max_acc]
        constr += [opt.abs(self.u[1, :]) <= self.vehicle.max_steer_rate]

        # Actuation rate of change bounds (step 0 uses last cmd)

        # constr += [
        #    opt.abs(self.u[0, 0] - self.last_cmd_param[0]) / self.dt
        #    <= self.vehicle.max_jerk
        # ]
        # for k in range(1, self.control_horizon):
        #    constr += [
        #        opt.abs(self.u[0, k] - self.u[0, k - 1]) / self.dt
        #        <= self.vehicle.max_jerk
        #    ]

        prob = opt.Problem(opt.Minimize(cost), constr)
        return prob

    def solve(
        self,
        initial_state: npt.NDArray[np.float64] | list[float],
        target: npt.NDArray[np.float64],
        verbose: bool = False,
        max_iter: int = 3,
        tolerance: float = 1e-2,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        assert len(initial_state) == self.nx
        assert target.shape == (self.nx, self.control_horizon + 1)

        self.initial_state_param.value = np.array(initial_state)
        self.last_cmd_param.value = (
            self.prev_cmd[:, 0] if self.prev_cmd is not None else np.zeros(self.nu)
        )

        # Extract references
        x_ref, y_ref = target[0, :], target[1, :]
        v_ref, theta_ref, delta_ref = target[2, :], target[3, :], target[4, :]

        # Pre-calculate scalar projections using np vectors
        cos_vals = np.cos(theta_ref)
        sin_vals = np.sin(theta_ref)
        p_along_vals = cos_vals * x_ref + sin_vals * y_ref
        p_cross_vals = -sin_vals * x_ref + cos_vals * y_ref

        # 4. Set values to vector parameters instantly
        self.cos_param.value = cos_vals
        self.sin_param.value = sin_vals
        self.p_along_ref_param.value = p_along_vals
        self.p_cross_ref_param.value = p_cross_vals
        self.v_ref_param.value = v_ref
        self.theta_ref_param.value = theta_ref
        self.delta_ref_param.value = delta_ref

        # To compute the system matrices for the LTV system, we may initially think to linearize the vehicle's nonlinear kinematics (like sin/cos/tan
        # steering math) **once** around the current state.
        # A, B, C = self.compute_linear_model_matrices(initial_state, prev_cmd)
        # It creates a flat tangent
        # line and assumes the vehicle physics will behave linearly for the next N steps.
        # This linear approximation gets more inaccurate as the controller looks at the future
        # , as the system changes (a lot!) along the trajectory, think sharp turns etc..
        # You will see the prediction is MUCH less accurate as the horizon grows...
        #
        #
        # In iMPC instead of linearizing once at the start, we make an initial guess of
        # the entire future trajectory and linearize at *every individual step* along that guessed path.
        #
        # After solving the optimization problem, we update the guessed trajectory,
        # re-linearizing around the new path, we repeat this up to N times.
        #
        # Eventually the linear models will converges onto the true, curved, non-linear physics of
        # the vehicle before a command is ever sent to the actuators.

        # Form the Initial Guess for the iMPC loop
        if self.prev_trajectory is not None and self.prev_cmd is not None:
            # Shift previous optimal trajectory left by 1 timestep
            x_guess = np.roll(self.prev_trajectory, -1, axis=1)
            x_guess[:, -1] = self.prev_trajectory[:, -1]
            u_guess = np.roll(self.prev_cmd, -1, axis=1)
            u_guess[:, -1] = self.prev_cmd[:, -1]
        else:
            # first iteration guess: pretend the vehicle follows the reference perfectly
            x_guess = target
            u_guess = np.zeros((self.nu, self.control_horizon))

        # The iMPC Optimization Loop
        for iteration in range(max_iter):
            # Linearize around our current best guess
            for k in range(self.control_horizon):
                x_bar = x_guess[:, k]
                u_bar = u_guess[:, k]

                A_k, B_k, C_k = self.compute_linear_model_matrices(x_bar, u_bar)
                self.A_params[k].value = A_k
                self.B_params[k].value = B_k
                self.C_params[k].value = C_k

            self.prob.solve(
                solver=opt.CLARABEL,
                warm_start=True,
                verbose=verbose,
                canon_backend=opt.SCIPY_CANON_BACKEND,
                enforce_dpp=True,
            )

            if self.x.value is None:
                # the oprimiser failed!
                # In this case you want to initialise a recovery behaviour!
                # to make this simple here I just decelerate
                print("MPC failed -> Emergency braking!")

                # wipe memory so the next loop cycle resets cleanly
                self.prev_trajectory = None
                self.prev_cmd = None

                emergency_u = np.zeros((self.nu, self.control_horizon))
                v = initial_state[2]
                for k in range(self.control_horizon):
                    a = -self.vehicle.max_acc if v > 0 else 0.0
                    emergency_u[0, k] = a
                    v = max(0.0, v + a * self.dt)
                self.prev_cmd = np.copy(emergency_u)
                return None, self.prev_cmd

            new_x = np.array(self.x.value)
            new_u = np.array(self.u.value)

            # If the maximum deviation between the old guess and the new solution is tiny,
            # the non-linear approximations have converged. Success.
            if np.max(np.abs(new_x - x_guess)) < tolerance:
                break

            # Update the guess for the next iteration
            x_guess = new_x
            u_guess = new_u

        # Store the finalized optimal trajectory for the next control cycle
        self.prev_trajectory = np.copy(new_x)
        self.prev_cmd = np.copy(new_u)

        return self.prev_trajectory, self.prev_cmd

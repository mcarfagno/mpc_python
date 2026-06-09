from __future__ import annotations

import pathlib

import cvxpy as opt
import numpy as np
import numpy.typing as npt
import yaml
import scipy.linalg


class MPC:
    def __init__(
        self,
        config: str | pathlib.Path | dict,
        horizon_time: float | None = None,
        timestep: float | None = None,
        state_cost: list[float] | None = None,
        final_state_cost: list[float] | None = None,
        input_cost: list[float] | None = None,
        input_rate_cost: list[float] | None = None,
        slack_penalty: float | None = None,
    ) -> None:
        if isinstance(config, (str, pathlib.Path)):
            # resolve path
            path = pathlib.Path(config)
            if not path.is_absolute():
                path = pathlib.Path(__file__).parent.parent / config
            with open(path) as f:
                config_data = yaml.safe_load(f)
        else:
            # raw dict
            config_data = config

        vehicle_config = config_data["model"]["vehicle"]
        obstacle_config = config_data["controller"]["obstacle"]
        prediction_config = config_data["controller"]["prediction"]
        weights_config = config_data["controller"]["weights"]

        self._state_dim: int = 4
        self._control_dim: int = 2

        self.wheelbase: float = vehicle_config["wheelbase"]
        self.width: float = vehicle_config["width"]
        self.max_speed: float = vehicle_config["max_speed"]
        self.max_acc: float = vehicle_config["max_acc"]
        self.max_d_acc: float = vehicle_config["max_d_acc"]
        self.max_steer: float = vehicle_config["max_steer"]
        self.max_d_steer: float = vehicle_config["max_d_steer"]

        self.dt: float = (
            timestep if timestep is not None else prediction_config["timestep"]
        )
        horizon = (
            horizon_time
            if horizon_time is not None
            else prediction_config["horizon_time"]
        )
        self.control_horizon: int = int(horizon / self.dt)

        state_cost_weights = (
            state_cost if state_cost is not None else weights_config["state_cost"]
        )
        terminal_cost_weights = (
            final_state_cost
            if final_state_cost is not None
            else weights_config["final_state_cost"]
        )
        input_cost_weights = (
            input_cost if input_cost is not None else weights_config["input_cost"]
        )
        input_rate_cost_weights = (
            input_rate_cost
            if input_rate_cost is not None
            else weights_config["input_rate_cost"]
        )

        if len(state_cost_weights) != self._state_dim:
            raise ValueError(
                f"State Error cost matrix should be of size {self._state_dim}"
            )
        if len(terminal_cost_weights) != self._state_dim:
            raise ValueError(
                f"End State Error cost matrix should be of size {self._state_dim}"
            )
        if len(input_cost_weights) != self._control_dim:
            raise ValueError(
                f"Control Effort cost matrix should be of size {self._control_dim}"
            )
        if len(input_rate_cost_weights) != self._control_dim:
            raise ValueError(
                "Control Effort Difference cost matrix should be of size "
                f"{self._control_dim}"
            )

        # self.q_matrix: npt.NDArray[np.float64] = np.diag(state_cost_weights)
        # self.qf_matrix: npt.NDArray[np.float64] = np.diag(terminal_cost_weights)
        # self.r_matrix: npt.NDArray[np.float64] = np.diag(input_cost_weights)
        # self.rr_matrix: npt.NDArray[np.float64] = np.diag(input_rate_cost_weights)

        # NOTE: we use sum_squares wich is not the same as a quad_form(x,Q)
        # To use sum_squares correctly, you need to pass it a matrix A such that A^T A = Q
        self.q_matrix: npt.NDArray[np.float64] = scipy.linalg.cholesky(
            np.diag(state_cost_weights)
        )
        self.qf_matrix: npt.NDArray[np.float64] = scipy.linalg.cholesky(
            np.diag(terminal_cost_weights)
        )
        self.r_matrix: npt.NDArray[np.float64] = scipy.linalg.cholesky(
            np.diag(input_cost_weights)
        )
        self.rr_matrix: npt.NDArray[np.float64] = scipy.linalg.cholesky(
            np.diag(input_rate_cost_weights)
        )

        self._safety_margin: float = obstacle_config["safety_margin"]
        self._slack_penalty: float = (
            slack_penalty
            if slack_penalty is not None
            else obstacle_config["slack_penalty"]
        )

        self._vehicle_buffer: float = self.width / 2.0 + self._safety_margin

        # CVXPY vars
        self._states: opt.Variable = opt.Variable(
            (self._state_dim, self.control_horizon + 1), name="states"
        )
        self._controls: opt.Variable = opt.Variable(
            (self._control_dim, self.control_horizon), name="actions"
        )

        # CVXPY params (placeholder for run-time data)
        self._initial_state: opt.Parameter = opt.Parameter(self._state_dim, name="x0")
        self._last_command: opt.Parameter = opt.Parameter(
            self._control_dim, name="last_applied_command"
        )

        self._A_params: list[opt.Parameter] = [
            opt.Parameter((self._state_dim, self._state_dim), name=f"A_{k}")
            for k in range(self.control_horizon)
        ]
        self._B_params: list[opt.Parameter] = [
            opt.Parameter((self._state_dim, self._control_dim), name=f"B_{k}")
            for k in range(self.control_horizon)
        ]
        self._C_params: list[opt.Parameter] = [
            opt.Parameter(self._state_dim, name=f"C_{k}")
            for k in range(self.control_horizon)
        ]

        # Reference params (DPP-compliant placeholders)
        # see https://www.cvxpy.org/tutorial/dpp/index.html
        self._cos_reference = opt.Parameter(self.control_horizon + 1)
        self._sin_reference = opt.Parameter(self.control_horizon + 1)
        self._along_reference = opt.Parameter(self.control_horizon + 1)
        self._cross_reference = opt.Parameter(self.control_horizon + 1)
        self._velocity_reference = opt.Parameter(self.control_horizon + 1)
        self._heading_reference = opt.Parameter(self.control_horizon + 1)

        # Obstacle params (half-plane linearization)
        self._obstacle_normal_x = opt.Parameter(self.control_horizon, name="obs_nx")
        self._obstacle_normal_y = opt.Parameter(self.control_horizon, name="obs_ny")
        self._obstacle_safe_distance = opt.Parameter(
            self.control_horizon, name="obs_dist"
        )
        # In optimization, a "slack" variable is a mathematical fudge factor.
        # Instead of treating the obstacle as an unyielding concrete wall,
        # we treat it as a stiff rubber wall. This variable tracks *how much*
        # we dent the wall if the vehicle is physically forced into it.
        # This allows in practice to turn the obstacle factor from hard(may cause failures) to soft
        self._obstacle_slack: opt.Variable = opt.Variable(
            self.control_horizon, nonneg=True, name="obstacle_slacks"
        )

        self._previous_command: npt.NDArray[np.float64] | None = None
        self._previous_trajectory: npt.NDArray[np.float64] | None = None

        # build the problem ONCE
        self._problem: opt.Problem = self._make_mpc_problem()

    def _compute_linear_model_matrices(
        self,
        x_bar: npt.NDArray[np.float64],
        u_bar: npt.NDArray[np.float64],
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        v = x_bar[2]
        theta = x_bar[3]

        a = u_bar[0]
        delta = u_bar[1]

        ct = np.cos(theta)
        st = np.sin(theta)
        cd = np.cos(delta)
        td = np.tan(delta)

        A = np.zeros((self._state_dim, self._state_dim))
        A[0, 2] = ct
        A[0, 3] = -v * st
        A[1, 2] = st
        A[1, 3] = v * ct
        A[3, 2] = v * td / self.wheelbase
        A_lin = np.eye(self._state_dim) + self.dt * A

        B = np.zeros((self._state_dim, self._control_dim))
        B[2, 0] = 1
        B[3, 1] = v / (self.wheelbase * cd**2)
        B_lin = self.dt * B

        f_xu = np.array([v * ct, v * st, a, v * td / self.wheelbase]).reshape(
            self._state_dim, 1
        )
        C_lin = (
            self.dt
            * (
                f_xu
                - np.dot(A, x_bar.reshape(self._state_dim, 1))
                - np.dot(B, u_bar.reshape(self._control_dim, 1))
            ).flatten()
        )
        return A_lin, B_lin, C_lin

    def _make_mpc_problem(self) -> opt.Problem:

        cost = 0
        constraints = []

        for k in range(self.control_horizon):
            # Kinematics constrains
            # Note each step uses the LTV matrix for that step
            constraints += [
                self._states[:, k + 1]
                == self._A_params[k] @ self._states[:, k]
                + self._B_params[k] @ self._controls[:, k]
                + self._C_params[k]
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
            along_track_error = (
                self._cos_reference[k] * self._states[0, k]
                + self._sin_reference[k] * self._states[1, k]
                - self._along_reference[k]
            )
            cross_track_error = (
                -self._sin_reference[k] * self._states[0, k]
                + self._cos_reference[k] * self._states[1, k]
                - self._cross_reference[k]
            )
            error = opt.vstack(
                [
                    along_track_error,
                    cross_track_error,
                    self._states[2, k] - self._velocity_reference[k],
                    self._states[3, k] - self._heading_reference[k],
                ]
            )
            cost += opt.sum_squares(self.q_matrix @ error)
            # Obstacle half-plane constraint:
            # obstacle avoidance: (px - pobs)*2 > R  in non-convex :(
            # this is linearised as : dot(px - pbos, n) > R
            # where n is a normal pointing from obstacle center toward the reference trajectory.
            # so the optimiser knows to stay on the same side of the obstacle as the reference
            #
            #      Valid Region
            #            ^
            #            | n = (nx, ny)
            #            * p_ref
            #            |
            #    --------+---------- Half-Plane
            #            | R
            #            * p_obs
            #
            # Hard Constraint:  dot(n, p) >= Safe_Distance
            # Soft Constraint:  dot(n, p) >= Safe_Distance - Slack
            constraints += [
                self._obstacle_normal_x[k] * self._states[0, k + 1]
                + self._obstacle_normal_y[k] * self._states[1, k + 1]
                >= self._obstacle_safe_distance[k] - self._obstacle_slack[k]
            ]
            cost += self._slack_penalty * self._obstacle_slack[k]

            cost += opt.sum_squares(self.r_matrix @ self._controls[:, k])
            if k == 0:
                cost += opt.sum_squares(
                    self.rr_matrix @ (self._controls[:, 0] - self._last_command)
                )
            else:
                cost += opt.sum_squares(
                    self.rr_matrix @ (self._controls[:, k] - self._controls[:, k - 1])
                )
        terminal_along_track_error = (
            self._cos_reference[-1] * self._states[0, -1]
            + self._sin_reference[-1] * self._states[1, -1]
            - self._along_reference[-1]
        )
        terminal_cross_track_error = (
            -self._sin_reference[-1] * self._states[0, -1]
            + self._cos_reference[-1] * self._states[1, -1]
            - self._cross_reference[-1]
        )
        terminal_error = opt.vstack(
            [
                terminal_along_track_error,
                terminal_cross_track_error,
                self._states[2, -1] - self._velocity_reference[-1],
                self._states[3, -1] - self._heading_reference[-1],
            ]
        )
        cost += opt.sum_squares(self.qf_matrix @ terminal_error)

        # initial state
        constraints += [self._states[:, 0] == self._initial_state]

        # state bounds
        constraints += [opt.abs(self._states[2, :]) <= self.max_speed]

        # actuation bounds
        constraints += [opt.abs(self._controls[0, :]) <= self.max_acc]
        constraints += [opt.abs(self._controls[1, :]) <= self.max_steer]
        for k in range(1, self.control_horizon):
            constraints += [
                opt.abs(self._controls[0, k] - self._controls[0, k - 1])
                <= self.max_d_acc * self.dt
            ]
            constraints += [
                opt.abs(self._controls[1, k] - self._controls[1, k - 1])
                <= self.max_d_steer * self.dt
            ]

        problem = opt.Problem(opt.Minimize(cost), constraints)
        return problem

    def solve(
        self,
        initial_state: npt.NDArray[np.float64] | list[float],
        target: npt.NDArray[np.float64],
        verbose: bool = False,
        max_iter: int = 3,
        tolerance: float = 1e-2,
        obstacle: tuple[float, float, float] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        assert len(initial_state) == self._state_dim
        assert target.shape == (self._state_dim, self.control_horizon + 1)

        (
            obstacle_x,
            obstacle_y,
            obstacle_radius,
            obstacle_velocity_x,
            obstacle_velocity_y,
        ) = (
            obstacle if obstacle is not None else (None, None, None, None, None)
        )

        self._initial_state.value = np.array(initial_state)
        self._last_command.value = (
            self._previous_command[:, 0]
            if self._previous_command is not None
            else np.zeros(self._control_dim)
        )

        x_ref, y_ref = target[0, :], target[1, :]
        v_ref, theta_ref = target[2, :], target[3, :]

        # Pre-calculate scalar projections using NumPy vectors
        cos_values = np.cos(theta_ref)
        sin_values = np.sin(theta_ref)
        along_projections = cos_values * x_ref + sin_values * y_ref
        cross_projections = -sin_values * x_ref + cos_values * y_ref

        self._cos_reference.value = cos_values
        self._sin_reference.value = sin_values
        self._along_reference.value = along_projections
        self._cross_reference.value = cross_projections
        self._velocity_reference.value = v_ref
        self._heading_reference.value = theta_ref

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
        if self._previous_trajectory is not None and self._previous_command is not None:
            # Shift previous optimal trajectory left by 1 timestep
            x_guess = np.roll(self._previous_trajectory, -1, axis=1)
            x_guess[:, -1] = self._previous_trajectory[:, -1]
            u_guess = np.roll(self._previous_command, -1, axis=1)
            u_guess[:, -1] = self._previous_command[:, -1]
        else:
            # first iteration guess: pretend the vehicle follows the reference perfectly
            x_guess = target
            u_guess = np.zeros((self._control_dim, self.control_horizon))

        # The iMPC Optimization Loop
        for _ in range(max_iter):
            for k in range(self.control_horizon):
                x_bar = x_guess[:, k]
                u_bar = u_guess[:, k]

                A_k, B_k, C_k = self._compute_linear_model_matrices(x_bar, u_bar)
                self._A_params[k].value = A_k
                self._B_params[k].value = B_k
                self._C_params[k].value = C_k

            # Obstacle half-plane params (based on x_guess at step k+1)
            obstacle_normals_x = np.zeros(self.control_horizon)
            obstacle_normals_y = np.zeros(self.control_horizon)
            obstacle_distances = np.zeros(self.control_horizon)
            for k in range(self.control_horizon):
                if obstacle is None:
                    # turn this in something trivial for the optimiser
                    obstacle_normals_x[k] = 1.0
                    obstacle_normals_y[k] = 0.0
                    obstacle_distances[k] = -1000.0
                else:
                    # We need a stable point to calculate the normal vector.
                    # Also we use the current belief of the obstacle moving velocity
                    # to predict where its position will be at each k of th horizon
                    dx = x_ref[k + 1] - obstacle_x - obstacle_velocity_x * k * self.dt
                    dy = y_ref[k + 1] - obstacle_y - obstacle_velocity_y * k * self.dt

                    dist = np.hypot(dx, dy)
                    dist = dist if dist > 1e-5 else 1e-5
                    # [x,y] components of vector n
                    normal_x = dx / dist
                    normal_y = dy / dist

                    obstacle_normals_x[k] = normal_x
                    obstacle_normals_y[k] = normal_y
                    obstacle_distances[k] = (
                        normal_x * obstacle_x + normal_y * obstacle_y + obstacle_radius
                    )

            self._obstacle_normal_x.value = obstacle_normals_x
            self._obstacle_normal_y.value = obstacle_normals_y
            self._obstacle_safe_distance.value = (
                obstacle_distances + self._vehicle_buffer
            )

            self._problem.solve(
                solver=opt.CLARABEL,
                enforce_dpp=True,
                warm_start=True,
                verbose=verbose,
                canon_backend=opt.SCIPY_CANON_BACKEND,
                tol_gap_abs=1e-3,  # Loosen absolute duality gap tolerance
                tol_gap_rel=1e-3,  # Loosen relative duality gap tolerance
                tol_feas=1e-3,  # Loosen feasibility tolerance
                max_iter=25,  # Hard cap on interior-point iterations
            )

            if self._states.value is None:
                # the optimiser failed!
                # In this case you want to initialise a recovery behaviour!
                # To make this simple here I just decelerate
                print("MPC failed -> Emergency braking!")
                emergency_controls = np.zeros((self._control_dim, self.control_horizon))
                v = initial_state[2]
                for k in range(self.control_horizon):
                    a = -self.max_acc if v > 0 else 0.0
                    emergency_controls[0, k] = a
                    v = max(0.0, v + a * self.dt)
                self._previous_command = np.copy(emergency_controls)
                return None, self._previous_command

            new_x = np.array(self._states.value)
            new_u = np.array(self._controls.value)

            # If the maximum deviation between the old guess and the new solution is tiny,
            # the non-linear approximations have converged. Success.
            if np.max(np.abs(new_x - x_guess)) < tolerance:
                break
            # Update the guess for the next iteration
            x_guess = new_x
            u_guess = new_u

        # Store the finalized optimal trajectory for the next control cycle
        self._previous_trajectory = np.copy(new_x)
        self._previous_command = np.copy(new_u)

        return self._previous_trajectory, self._previous_command

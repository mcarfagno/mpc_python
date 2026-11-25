from __future__ import annotations

import pathlib

import cvxpy as opt
import numpy as np
import numpy.typing as npt
import yaml


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

        # NOTE: we use sum_squares wich is not the same as a quad_form(x,Q), this is for strict DDP compliance
        # But to use sum_squares correctly, you need to pass it a matrix A such that A^T A = Q
        # We can get away with sqrt because matrices are diagonal
        self.q_matrix: npt.NDArray[np.float64] = np.sqrt(np.diag(state_cost_weights))
        self.qf_matrix: npt.NDArray[np.float64] = np.sqrt(
            np.diag(terminal_cost_weights)
        )
        self.r_matrix: npt.NDArray[np.float64] = np.sqrt(np.diag(input_cost_weights))
        self.rr_matrix: npt.NDArray[np.float64] = np.sqrt(
            np.diag(input_rate_cost_weights)
        )

        self._safety_margin: float = obstacle_config["safety_margin"]
        self._slack_penalty: float = (
            slack_penalty
            if slack_penalty is not None
            else obstacle_config["slack_penalty"]
        )

        self._vehicle_buffer: float = self.width / 2.0 + self._safety_margin
        track_config = config_data["controller"].get("track", {})
        self.track_half_width: float = float(track_config.get("half_width") or 0.0)
        self._track_inner: float = max(
            self.track_half_width - self._vehicle_buffer, 0.0
        )
        self._track_slack_penalty: float = float(
            track_config.get("slack_penalty", self._slack_penalty)
        )

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
        self._obstacle_side: float = 1.0  # +1 = pass left when the obstacle center is on the path
        # One slack for the whole lane. Reuses the cross-track error, so the
        # corridor adds no parameters and one variable.
        self._track_slack: opt.Variable | None = (
            opt.Variable(nonneg=True, name="track_slack")
            if self.track_half_width > 0.0
            else None
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
        A[3, 2] = td / self.wheelbase
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
            # The standard cross-track and along-track errors are calculated
            # by projecting position errors onto the track point:
            # Here is the CTE:
            #                   + e
            #                   |
            #                   n = (-sinθ, cosθ)
            #                   |
            #                   * p  (vehicle)
            #                  /|
            #                 / |  e = cross-track error
            #                /  |
            #   ============*---+---------------->  t = (cosθ, sinθ)
            #              p_ref
            #                   |
            #                   - e
            #
            #
            # So e = n · (p - p_ref)  = -sinθ (x - x_ref) + cosθ (y - y_ref)
            # e = -sinθ (x - x_ref) + cosθ (y - y_ref)
            #   = [-sinθ · x + cosθ · y]   -   [-sinθ · x_ref + cosθ · y_ref]
            #     ─────────────────────         ─────────────────────────────
            #     states (decision vars)        _cross_reference[k], a parameter
            #                                   precomputed once in solve()
            # where + to the left of the path, - to the right


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

            # The corridor is the cross-track error kept inside the lane.
            if self._track_slack is not None:
                constraints += [
                    cross_track_error <= self._track_inner + self._track_slack,
                    cross_track_error >= -self._track_inner - self._track_slack,
                ]

            cost += opt.sum_squares(self.r_matrix @ self._controls[:, k])
            if k == 0:
                cost += opt.sum_squares(
                    self.rr_matrix @ (self._controls[:, 0] - self._last_command)
                )
            else:
                cost += opt.sum_squares(
                    self.rr_matrix @ (self._controls[:, k] - self._controls[:, k - 1])
                )

        if self._track_slack is not None:
            cost += self._track_slack_penalty * self._track_slack
        
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
        if self._track_slack is not None:
            constraints += [
                terminal_cross_track_error <= self._track_inner + self._track_slack,
                terminal_cross_track_error
                >= -self._track_inner - self._track_slack,
            ]
        cost += opt.sum_squares(self.qf_matrix @ terminal_error)

        # initial state
        constraints += [self._states[:, 0] == self._initial_state]

        # state bounds
        constraints += [opt.abs(self._states[2, :]) <= self.max_speed]

        # actuation bounds
        constraints += [opt.abs(self._controls[0, :]) <= self.max_acc]
        constraints += [opt.abs(self._controls[1, :]) <= self.max_steer]
        constraints += [
            opt.abs(self._controls[0, 0] - self._last_command[0])
            <= self.max_d_acc * self.dt
        ]
        constraints += [
            opt.abs(self._controls[1, 0] - self._last_command[1])
            <= self.max_d_steer * self.dt
        ]
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
        obstacle: tuple[float, float, float, float, float] | None = None,
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

        # Obstacle half-planes. They only depend on the reference and the
        # obstacle, not on the linearization guess, so fill them once here.
        # With no obstacle the plane is far away and the constraint is trivial.
        normal_x = np.ones(self.control_horizon)
        normal_y = np.zeros(self.control_horizon)
        distances = np.full(self.control_horizon, -1000.0)
        if obstacle is not None:
            ks = np.arange(1, self.control_horizon + 1)
            obs_xs = obstacle_x + obstacle_velocity_x * ks * self.dt
            obs_ys = obstacle_y + obstacle_velocity_y * ks * self.dt
            # Pick the passing side at the horizon step closest to the obstacle;
            # fall back to the vehicle's current side if the obstacle sits on the centerline.
            k_obs = int(np.argmin(np.hypot(x_ref[1:] - obs_xs, y_ref[1:] - obs_ys)))
            obs_lat = -sin_values[k_obs + 1] * (
                obs_xs[k_obs] - x_ref[k_obs + 1]
            ) + cos_values[k_obs + 1] * (obs_ys[k_obs] - y_ref[k_obs + 1])
            if abs(obs_lat) >= 0.05:
                self._obstacle_side = -1.0 if obs_lat > 0.0 else 1.0
            else:
                lat = (
                    -sin_values[0] * (initial_state[0] - obstacle_x)
                    + cos_values[0] * (initial_state[1] - obstacle_y)
                )
                if abs(lat) >= 0.05:
                    self._obstacle_side = 1.0 if lat > 0.0 else -1.0

            for k in range(self.control_horizon):
                # The constraint acts on state k+1, so predict the obstacle there.
                obs_x = obs_xs[k]
                obs_y = obs_ys[k]
                dx = x_ref[k + 1] - obs_x
                dy = y_ref[k + 1] - obs_y
                ct, st = cos_values[k + 1], sin_values[k + 1]
                # Decompose (ref - obs) in path frame and keep a minimum lateral
                # bias on the chosen side so normals stay continuous without flipping.
                v_along = dx * ct + dy * st
                v_lat = -dx * st + dy * ct
                v_lat = self._obstacle_side * max(self._obstacle_side * v_lat, 0.15)
                nx_raw = v_along * ct - v_lat * st
                ny_raw = v_along * st + v_lat * ct
                norm = np.hypot(nx_raw, ny_raw)
                nx = nx_raw / norm
                ny = ny_raw / norm
                normal_x[k] = nx
                normal_y[k] = ny
                distances[k] = nx * obs_x + ny * obs_y + obstacle_radius

        self._obstacle_normal_x.value = normal_x
        self._obstacle_normal_y.value = normal_y
        self._obstacle_safe_distance.value = distances + self._vehicle_buffer

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

            self._problem.solve(
                solver=opt.CLARABEL,
                enforce_dpp=True,
                warm_start=True,
                verbose=verbose,
                canon_backend=opt.SCIPY_CANON_BACKEND,
                tol_gap_abs=1e-3,  # relax tolerances
                tol_gap_rel=1e-3,
                tol_feas=1e-3,
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

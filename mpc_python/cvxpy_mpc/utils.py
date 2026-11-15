import numpy as np
import numpy.typing as npt
from scipy.interpolate import splprep, splev


def compute_path_from_wp(
    start_xp: list[float] | npt.NDArray[np.float64],
    start_yp: list[float] | npt.NDArray[np.float64],
    step: float = 0.1,
) -> npt.NDArray[np.float64]:
    """
    Generates a physically drivable, smooth C2 continuous path.

    Args:
        start_xp: X-coordinates of waypoints.
        start_yp: Y-coordinates of waypoints.
        step: Arc-length step between consecutive output points.

    Returns:
        Array of shape (3, N) with rows [x, y, heading] along the path.
    """
    # Fit a cubic spline (B-spline) to the waypoints
    # s=0 forces the spline to pass exactly through your waypoints.
    tck, u = splprep([start_xp, start_yp], s=0.0)

    # increase resolution to calculate arc length accurately
    u_fine = np.linspace(0, 1, 2000)
    x_fine, y_fine = splev(u_fine, tck)

    arc_lengths = np.zeros(len(u_fine))
    arc_lengths[1:] = np.cumsum(np.hypot(np.diff(x_fine), np.diff(y_fine)))
    total_len = arc_lengths[-1]

    # interpolate by step size
    num_points = int(total_len / step)
    u_uniform = np.interp(np.linspace(0, total_len, num_points), arc_lengths, u_fine)
    final_xp, final_yp = splev(u_uniform, tck)

    dx = np.gradient(final_xp)
    dy = np.gradient(final_yp)
    theta = np.arctan2(dy, dx)

    return np.vstack((final_xp, final_yp, theta))


def get_nn_idx(state: npt.NDArray[np.float64], path: npt.NDArray[np.float64]) -> int:
    """
    Finds the index of the closest path point to the vehicle, with forward projection.

    Args:
        state: Vehicle state [x, y, ...]. Only the first two elements are used.
        path: Array of shape (2, N) of [x, y] path points.

    Returns:
        Index of the closest or next-forward path point.
    """
    dx = state[0] - path[0, :]
    dy = state[1] - path[1, :]
    dist = np.hypot(dx, dy)
    nn_idx = np.argmin(dist)
    try:
        v = np.array(
            [
                path[0, nn_idx + 1] - path[0, nn_idx],
                path[1, nn_idx + 1] - path[1, nn_idx],
            ]
        )
        assert np.linalg.norm(v) > 0, "zero-length path segment"
        v /= np.linalg.norm(v)
        d = [path[0, nn_idx] - state[0], path[1, nn_idx] - state[1]]
        if np.dot(d, v) > 0:
            target_idx = nn_idx
        else:
            target_idx = nn_idx + 1
    except IndexError:
        target_idx = nn_idx
    return target_idx


def get_ref_trajectory(
    state: npt.NDArray[np.float64],
    path: npt.NDArray[np.float64],
    target_v: float,
    T: float,
    DT: float,
    ego_frame: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Builds a reference trajectory from the path for the MPC horizon.

    Args:
        state: Vehicle state in global frame [x, y, v, heading].
        path: Array of shape (3, N) with rows [x, y, heading] in global frame.
        target_v: Desired forward speed.
        T: Control horizon duration.
        DT: Control horizon time-step.
        ego_frame: If True, returns trajectory in the vehicle's ego frame.
            If False, returns in the global world frame.

    Returns:
        Array of shape (4, K+1) with rows [x, y, v, heading] representing
        the reference trajectory over the horizon.
    """
    K = int(T / DT)

    # Allocate K + 1 elements to map exactly from k=0 (initial) to k=K (terminal)
    xref = np.zeros((4, K + 1))
    ind = get_nn_idx(state, path)

    # Calculate cumulative distance along the path
    cdist = np.append(
        [0.0], np.cumsum(np.hypot(np.diff(path[0, :]), np.diff(path[1, :])))
    )
    cdist = np.clip(cdist, cdist[0], cdist[-1])

    start_dist = cdist[ind]

    # range is (0, K + 1) to include the t=0 starting node
    interp_points = [d * DT * target_v + start_dist for d in range(0, K + 1)]

    # Compute interpolation (automatically maps across all K + 1 points)
    xref[0, :] = np.interp(interp_points, cdist, path[0, :])
    xref[1, :] = np.interp(interp_points, cdist, path[1, :])
    xref[2, :] = target_v
    xref[3, :] = np.interp(interp_points, cdist, path[2, :])

    xref_cdist = np.interp(interp_points, cdist, cdist)
    stop_idx = np.where(xref_cdist == cdist[-1])
    xref[2, stop_idx] = 0.0

    if ego_frame:
        dx = xref[0, :] - state[0]
        dy = xref[1, :] - state[1]
        xref[0, :] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])  # Local X
        xref[1, :] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])  # Local Y

        xref[3, :] = xref[3, :] - state[3]  # Local Theta

    # Continuous angle smoothing
    def fix_angle_reference(angle_ref, angle_init):
        diff_angle = angle_ref - angle_init
        diff_angle = np.unwrap(diff_angle)
        return angle_init + diff_angle

    # Normalize to [-pi, pi], then unwrap to remove any discontinuity
    xref[3, :] = (xref[3, :] + np.pi) % (2.0 * np.pi) - np.pi
    xref[3, :] = fix_angle_reference(xref[3, :], xref[3, 0])

    return xref


def ego_to_global(
    state: npt.NDArray[np.float64], x_mpc: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Transforms MPC trajectory from ego frame to global frame.

    Args:
        state: Vehicle state [x, y, v, heading] in global frame.
        x_mpc: MPC solution trajectory in ego frame (2, N).

    Returns:
        Trajectory in global frame (2, N).
    """
    traj = x_mpc[:2, :].copy()
    ct, st = np.cos(state[3]), np.sin(state[3])
    R = np.array([[ct, -st], [st, ct]])
    traj = R @ traj
    traj[0, :] += state[0]
    traj[1, :] += state[1]
    return traj

def obstacle_to_ego(
    state: npt.NDArray[np.float64],
    obstacle: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float, float]:
    """Transform (x, y, r, vx, vy) from global frame into the vehicle ego frame.

    Args:
        state: Vehicle state [x, y, v, heading] in global frame.
        obstacle: Obstacle state [x, y, r, vx, vy] in global frame.

    Returns
        Obstacle state [x, y, r, vx, vy] in ego vehicle state.
    """
    gx, gy, r, vx, vy = obstacle
    dx, dy = gx - state[0], gy - state[1]
    ct, st = np.cos(-state[3]), np.sin(-state[3])
    return (
        dx * ct - dy * st,
        dy * ct + dx * st,
        r,
        vx * ct - vy * st,
        vy * ct + vx * st,
    )

def compute_path_arc_lengths(
    path: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float]:
    """Compute cumulative arc-length along a (3,N) path.

    Args:
        path: Array of shape (3, N) with rows [x, y, heading].

    Returns:
        Tuple of (cdist, total_length) where cdist is the cumulative
        arc-length array and total_length is the total path length.
    """
    cdist = np.zeros(path.shape[1])
    cdist[1:] = np.cumsum(np.hypot(np.diff(path[0]), np.diff(path[1])))
    return cdist, cdist[-1]


def update_path_obstacles(
    obstacles: list[dict],
    path: npt.NDArray[np.float64],
    dt: float,
) -> list[list[float]]:
    """Advance path-following obstacles and return [x, y, radius, vx, vy] list.

    Each obstacle dict: {"distance": float, "speed": float, "radius": float}
    Obstacles wrap around at the path end.
    """
    # ideally this should only be precomputed once
    cdist, total_length = compute_path_arc_lengths(path)

    result = []
    for obs in obstacles:
        obs["distance"] = (obs["distance"] + obs["speed"] * dt) % total_length
        x = np.interp(obs["distance"], cdist, path[0])
        y = np.interp(obs["distance"], cdist, path[1])

        idx = max(
            0,
            min(
                np.searchsorted(cdist, obs["distance"]) - 1,
                path.shape[1] - 2,
            ),
        )
        seg_dx = path[0, idx + 1] - path[0, idx]
        seg_dy = path[1, idx + 1] - path[1, idx]
        seg_len = np.hypot(seg_dx, seg_dy)
        if seg_len > 1e-6:
            tx = seg_dx / seg_len
            ty = seg_dy / seg_len
            vx = tx * obs["speed"]
            vy = ty * obs["speed"]
            lateral = obs.get("lateral_offset", 0.0)
            if lateral != 0.0:
                x += -ty * lateral
                y += tx * lateral
        else:
            vx, vy = 0.0, 0.0

        result.append([x, y, obs["radius"], vx, vy])
    return result


def compute_errors(
    current_state: npt.NDArray[np.float64], path: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Compute signed cross-track error and heading error.

    Signed_cte is positive if vehicle is to the left.
    """
    assert path.shape[1] >= 2, "path must have at least 2 points"
    # Find the closest waypoint index
    dx = current_state[0] - path[0, :]
    dy = current_state[1] - path[1, :]
    distances = np.hypot(dx, dy)
    idx = np.argmin(distances)

    # Determine segment direction for true cross-track projection
    # If we are at the very last point, look backward, otherwise look forward
    if idx == path.shape[1] - 1:
        idx_start = idx - 1
        idx_end = idx
    else:
        idx_start = idx
        idx_end = idx + 1
    # Calculate forward-facing tangent vector
    tx = path[0, idx_end] - path[0, idx_start]
    ty = path[1, idx_end] - path[1, idx_start]
    seg_len = np.hypot(tx, ty)

    if seg_len > 1e-5:
        # Normalize tangent vector
        tx /= seg_len
        ty /= seg_len

        # Vector from waypoint to vehicle
        vx = current_state[0] - path[0, idx]
        vy = current_state[1] - path[1, idx]

        # True Cross-Track Error is the perpendicular scalar projection (Using 2D Cross Product)
        cte = (vy * tx) - (vx * ty)
    else:
        cte = distances[idx]

    # Heading Error (Normalized between -pi and pi)
    target_heading = path[2, idx_start]
    heading_err = (current_state[3] - target_heading + np.pi) % (2.0 * np.pi) - np.pi

    return (cte, heading_err)


def detect_obstacle_camera(
    obstacles: list[tuple[float, float, float, float, float]],
    robot_x: float,
    robot_y: float,
    robot_heading: float,
    max_range: float,
    fov_degrees: float = 60.0,
) -> tuple[float, float, float, float, float] | None:
    """Returns the closest obstacle within the robot's field of view.

    Args:
        obstacles: List of obstacles as [x, y, radius, vx, vy].
        robot_x: Robot x position.
        robot_y: Robot y position.
        robot_heading: Robot heading in radians.
        max_range: Maximum detection range.
        fov_degrees: Field of view in degrees (default 60).

    Returns:
        The closest obstacle tuple (x, y, radius, vx, vy) or None if no obstacle
        is within the FOV.
    """
    closest = None
    closest_dist = float("inf")
    fov_rad = np.radians(fov_degrees)
    for obs in obstacles:
        obs_x, obs_y, obs_r, obs_vx, obs_vy = obs

        dx = obs_x - robot_x
        dy = obs_y - robot_y
        d = np.hypot(dx, dy)

        dist_to_edge = max(0.0, d - obs_r)

        if dist_to_edge > max_range:
            continue

        # Normalize the relative angle to [-pi, pi]
        rel_angle = (np.arctan2(dy, dx) - robot_heading + np.pi) % (2.0 * np.pi) - np.pi
        angle_to_obs_center = abs(rel_angle)
        angular_radius = np.arcsin(obs_r / d)

        # Check if the closest edge of the obstacle falls within half the FOV
        if (angle_to_obs_center - angular_radius) <= (fov_rad / 2.0):

            # We track the closest obstacle by its closest EDGE, not its center
            if dist_to_edge < closest_dist:
                closest_dist = dist_to_edge
                closest = obs

    return closest

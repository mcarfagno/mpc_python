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
    Finds the index of the closest element

    Args:
        state (array-like): 1D array whose first two elements are x-pos and y-pos
        path (ndarray): 2D array of shape (2,N) of x,y points

    Returns:
        int: the index of the closest element
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
    except IndexError as e:
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
    Args:
        state (array-like): state of the vehicle in global frame [x, y, v, heading]
        path (ndarray): 2D array representing the path as x,y,heading points in global frame
        target_v (float): desired speed
        T (float): control horizon duration
        DT (float):  control horizon time-step
        ego_frame (bool): If True, returns trajectory in vehicle's ego frame.
                          If False, returns trajectory in the global world frame.

    Returns:
        ndarray: 2D array representing state space trajectory [x_k, y_k, v_k, theta_k]
                 with shape (4, K + 1) matching nodes from k=0 to k=K.
    """
    K = int(T / DT)

    # FIX 1: Allocate K + 1 elements to map exactly from k=0 (initial) to k=K (terminal)
    xref = np.zeros((5, K + 1))
    ind = get_nn_idx(state, path)

    # Calculate cumulative distance along the path
    cdist = np.append(
        [0.0], np.cumsum(np.hypot(np.diff(path[0, :]), np.diff(path[1, :])))
    )
    cdist = np.clip(cdist, cdist[0], cdist[-1])

    start_dist = cdist[ind]

    # FIX 2: Change range from (1, K + 1) to (0, K + 1) to include the t=0 starting node
    interp_points = [d * DT * target_v + start_dist for d in range(0, K + 1)]

    # Compute interpolation (automatically maps across all K + 1 points)
    xref[0, :] = np.interp(interp_points, cdist, path[0, :])
    xref[1, :] = np.interp(interp_points, cdist, path[1, :])
    xref[2, :] = target_v
    xref[3, :] = np.interp(interp_points, cdist, path[2, :])
    xref[4, :] = 0.0  # steer is usually zero

    xref_cdist = np.interp(interp_points, cdist, cdist)
    stop_idx = np.where(xref_cdist == cdist[-1])
    xref[2, stop_idx] = 0.0

    if ego_frame:
        dx = xref[0, :] - state[0]
        dy = xref[1, :] - state[1]
        xref[0, :] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])  # Local X
        xref[1, :] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])  # Local Y

        xref[3, :] = xref[3, :] - state[3]  # Local Theta

    # Continuous Angle Smoothing
    def fix_angle_reference(angle_ref, angle_init):
        diff_angle = angle_ref - angle_init
        diff_angle = np.unwrap(diff_angle)
        return angle_init + diff_angle

    xref[3, :] = (xref[3, :] + np.pi) % (2.0 * np.pi) - np.pi

    xref[3, :] = fix_angle_reference(xref[3, :], xref[3, 0])

    return xref


def ego_to_global(
    state: npt.NDArray[np.float64], x_mpc: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    traj = x_mpc[:2, :].copy()
    ct, st = np.cos(state[3]), np.sin(state[3])
    R = np.array([[ct, -st], [st, ct]])
    traj = R @ traj
    traj[0, :] += state[0]
    traj[1, :] += state[1]
    return traj


def compute_errors(
    current_state: npt.NDArray[np.float64], path: npt.NDArray[np.float64]
) -> tuple[float, float]:
    assert path.shape[1] >= 2, "path must have at least 2 points"
    # 1. Find the closest waypoint index
    dx = current_state[0] - path[0, :]
    dy = current_state[1] - path[1, :]
    distances = np.hypot(dx, dy)
    idx = np.argmin(distances)

    # 2. Determine segment direction for true cross-track projection
    # If we are at the very last point, look backward, otherwise look forward
    idx_next = idx - 1 if idx == path.shape[1] - 1 else idx + 1

    # Segment vector (tangent of the track)
    tx = path[0, idx_next] - path[0, idx]
    ty = path[1, idx_next] - path[1, idx]
    seg_len = np.hypot(tx, ty)

    if seg_len > 1e-5:
        # Normalize tangent vector
        tx /= seg_len
        ty /= seg_len

        # Vector from waypoint to vehicle
        vx = current_state[0] - path[0, idx]
        vy = current_state[1] - path[1, idx]

        # True Cross-Track Error is the perpendicular scalar projection (Using 2D Cross Product)
        cte = np.abs(vx * ty - vy * tx)
    else:
        cte = distances[idx]

    # 3. Heading Error (Normalized between -pi and pi)
    target_heading = path[2, idx]
    heading_err = (current_state[3] - target_heading + np.pi) % (2.0 * np.pi) - np.pi

    return (cte, heading_err)

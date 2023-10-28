import numpy as np
from scipy.interpolate import interp1d


def compute_path_from_wp(start_xp, start_yp, step=0.1):
    """

    Args:
        start_xp (array-like): 1D array of x-positions
        start_yp (array-like): 1D array of y-positions
        step (float): intepolation step

    Returns:
        ndarray of shape (3,N) representing the  path as x,y,heading
    """
    final_xp = []
    final_yp = []
    delta = step  # [m]
    for idx in range(len(start_xp) - 1):
        section_len = np.sum(
            np.sqrt(
                np.power(np.diff(start_xp[idx : idx + 2]), 2)
                + np.power(np.diff(start_yp[idx : idx + 2]), 2)
            )
        )
        interp_range = np.linspace(0, 1, np.floor(section_len / delta).astype(int))
        fx = interp1d(np.linspace(0, 1, 2), start_xp[idx : idx + 2], kind=1)
        fy = interp1d(np.linspace(0, 1, 2), start_yp[idx : idx + 2], kind=1)
        # watch out to duplicate points!
        final_xp = np.append(final_xp, fx(interp_range)[1:])
        final_yp = np.append(final_yp, fy(interp_range)[1:])
    dx = np.append(0, np.diff(final_xp))
    dy = np.append(0, np.diff(final_yp))
    theta = np.arctan2(dy, dx)
    return np.vstack((final_xp, final_yp, theta))


def get_nn_idx(state, path):
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
        v = [
            path[0, nn_idx + 1] - path[0, nn_idx],
            path[1, nn_idx + 1] - path[1, nn_idx],
        ]
        v /= np.linalg.norm(v)
        d = [path[0, nn_idx] - state[0], path[1, nn_idx] - state[1]]
        if np.dot(d, v) > 0:
            target_idx = nn_idx
        else:
            target_idx = nn_idx + 1
    except IndexError as e:
        target_idx = nn_idx
    return target_idx


def get_ref_trajectory(state, path, target_v, T, DT):
    """

    Args:
        state (array-like): state of the vehicle in global frame
        path (ndarray): 2D array representing the path as x,y,heading points in global frame
        target_v (float): desired speed
        T (float): control horizon duration
        DT (float):  control horizon time-step

    Returns:
        ndarray: 2D array representing state space trajectory [x_k, y_k, v_k, theta_k] w.r.t ego frame.
        Interpolated according to the time-step and the desired velocity
    """
    K = int(T / DT)

    xref = np.zeros((4, K))
    ind = get_nn_idx(state, path)

    cdist = np.append(
        [0.0], np.cumsum(np.hypot(np.diff(path[0, :].T), np.diff(path[1, :]).T))
    )
    cdist = np.clip(cdist, cdist[0], cdist[-1])

    start_dist = cdist[ind]
    interp_points = [d * DT * target_v + start_dist for d in range(1, K + 1)]
    xref[0, :] = np.interp(interp_points, cdist, path[0, :])
    xref[1, :] = np.interp(interp_points, cdist, path[1, :])
    xref[2, :] = target_v
    xref[3, :] = np.interp(interp_points, cdist, path[2, :])

    # points where the vehicle is at the end of trajectory
    xref_cdist = np.interp(interp_points, cdist, cdist)
    stop_idx = np.where(xref_cdist == cdist[-1])
    xref[2, stop_idx] = 0.0

    # transform in ego frame
    dx = xref[0, :] - state[0]
    dy = xref[1, :] - state[1]
    xref[0, :] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])  # X
    xref[1, :] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])  # Y
    xref[3, :] = path[2, ind] - state[3]  # Theta

    def fix_angle_reference(angle_ref, angle_init):
        """
        Removes jumps greater than 2PI to smooth the heading

        Args:
            angle_ref (array-like):
            angle_init (float):

        Returns:
            array-like:
        """
        diff_angle = angle_ref - angle_init
        diff_angle = np.unwrap(diff_angle)
        return angle_init + diff_angle

    xref[3, :] = (xref[3, :] + np.pi) % (2.0 * np.pi) - np.pi
    xref[3, :] = fix_angle_reference(xref[3, :], xref[3, 0])

    return xref

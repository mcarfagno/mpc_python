import numpy as np
from scipy.interpolate import interp1d
from .mpc_config import Params

P = Params()


def compute_path_from_wp(start_xp, start_yp, step=0.1):
    """
    Computes a reference path given a set of waypoints
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
    Computes the index of the waypoint closest to vehicle
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


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def get_ref_trajectory(state, path, target_v, dl=0.1):
    """
    For each step in the time horizon
    modified reference in robot frame
    """
    xref = np.zeros((P.N, P.T + 1))
    dref = np.zeros((1, P.T + 1))
    # sp = np.ones((1,T +1))*target_v #speed profile
    ncourse = path.shape[1]
    ind = get_nn_idx(state, path)
    dx = path[0, ind] - state[0]
    dy = path[1, ind] - state[1]
    xref[0, 0] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])  # X
    xref[1, 0] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])  # Y
    xref[2, 0] = target_v  # V
    xref[3, 0] = normalize_angle(path[2, ind] - state[3])  # Theta
    dref[0, 0] = 0.0  # Steer operational point should be 0
    travel = 0.0
    for i in range(1, P.T + 1):
        travel += abs(target_v) * P.DT
        dind = int(round(travel / dl))
        if (ind + dind) < ncourse:
            dx = path[0, ind + dind] - state[0]
            dy = path[1, ind + dind] - state[1]
            xref[0, i] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])
            xref[1, i] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])
            xref[2, i] = target_v  # sp[ind + dind]
            xref[3, i] = normalize_angle(path[2, ind + dind] - state[3])
            dref[0, i] = 0.0
        else:
            dx = path[0, ncourse - 1] - state[0]
            dy = path[1, ncourse - 1] - state[1]
            xref[0, i] = dx * np.cos(-state[3]) - dy * np.sin(-state[3])
            xref[1, i] = dy * np.cos(-state[3]) + dx * np.sin(-state[3])
            xref[2, i] = 0.0  # stop? #sp[ncourse - 1]
            xref[3, i] = normalize_angle(path[2, ncourse - 1] - state[3])
            dref[0, i] = 0.0
    return xref, dref

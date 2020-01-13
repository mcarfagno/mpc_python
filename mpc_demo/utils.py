import numpy as np
from scipy.interpolate import interp1d

def compute_path_from_wp(start_xp, start_yp, step = 0.1):
    """
    Interpolation range is computed to assure one point every fixed distance step [m].

    :param start_xp: array_like, list of starting x coordinates
    :param start_yp: array_like, list of starting y coordinates
    :param step: float, interpolation distance [m] between consecutive waypoints
    :returns: array_like, of shape (3,N)
    """

    final_xp=[]
    final_yp=[]
    delta = step #[m]

    for idx in range(len(start_xp)-1):
        section_len = np.sum(np.sqrt(np.power(np.diff(start_xp[idx:idx+2]),2)+np.power(np.diff(start_yp[idx:idx+2]),2)))

        interp_range = np.linspace(0,1,section_len/delta)

        fx=interp1d(np.linspace(0,1,2),start_xp[idx:idx+2],kind=1)
        fy=interp1d(np.linspace(0,1,2),start_yp[idx:idx+2],kind=1)

        final_xp=np.append(final_xp,fx(interp_range))
        final_yp=np.append(final_yp,fy(interp_range))

    dx = np.append(0, np.diff(final_xp))
    dy = np.append(0, np.diff(final_yp))
    theta = np.arctan2(dy, dx)

    return np.vstack((final_xp,final_yp,theta))

import numpy as np
from scipy.interpolate import interp1d

def compute_path_from_wp(start_xp, start_yp, step = 0.1):
    """
    """
    final_xp=[]
    final_yp=[]
    delta = step #[m]

    for idx in range(len(start_xp)-1):
        section_len = np.sqrt(np.sum(np.power(np.diff(start_xp[idx:idx+2]),2)+np.power(np.diff(start_yp[idx:idx+2]),2)))
        interp_range = np.linspace(0,1, int(1+section_len/delta))

        fx=interp1d(np.linspace(0,1,2),start_xp[idx:idx+2],kind=1)
        fy=interp1d(np.linspace(0,1,2),start_yp[idx:idx+2],kind=1)

        final_xp=np.append(final_xp,fx(interp_range))
        final_yp=np.append(final_yp,fy(interp_range))

    return np.vstack((final_xp,final_yp))

def get_nn_idx(state,path):
    """
    """
    dx = state[0]-path[0,:]
    dy = state[1]-path[1,:]
    dist = np.sqrt(dx**2 + dy**2)
    nn_idx = np.argmin(dist)

    try:
        v = [path[0,nn_idx+1] - path[0,nn_idx],
             path[1,nn_idx+1] - path[1,nn_idx]]
        v /= np.linalg.norm(v)

        d = [path[0,nn_idx] - state[0],
             path[1,nn_idx] - state[1]]

        if np.dot(d,v) > 0:
            target_idx = nn_idx
        else:
            target_idx = nn_idx+1

    except IndexError as e:
        target_idx = nn_idx

    return target_idx

def road_curve(state,track):
    """
    """

    POLY_RANK = 3

    #given vehicle pos find lookahead waypoints
    nn_idx=get_nn_idx(state,track)
    LOOKAHED = POLY_RANK*2
    lk_wp=track[:,max(0,nn_idx-1):nn_idx+LOOKAHED]

    #trasform lookahead waypoints to vehicle ref frame
    dx = lk_wp[0,:] - state[0]
    dy = lk_wp[1,:] - state[1]

    wp_vehicle_frame = np.vstack(( dx * np.cos(-state[3]) - dy * np.sin(-state[3]),
                                   dy * np.cos(-state[3]) + dx * np.sin(-state[3]) ))

    #fit poly
    return np.polyfit(wp_vehicle_frame[0,:], wp_vehicle_frame[1,:], POLY_RANK, rcond=None, full=False, w=None, cov=False)

# def f(x,coeff):
#     """
#     """
#     return round(coeff[0]*x**3 + coeff[1]*x**2 + coeff[2]*x**1 + coeff[3]*x**0,6)

# def f(x,coeff):
#     return  round(coeff[0]*x**5+coeff[1]*x**4+coeff[2]*x**3+coeff[3]*x**2+coeff[4]*x**1+coeff[5]*x**0,6)

def f(x,coeff):
    y=0
    j=len(coeff)
    for k in range(j):
        y += coeff[k]*x**(j-k-1)
    return round(y,6)

# def df(x,coeff):
#     """
#     """
#     return round(3*coeff[0]*x**2 + 2*coeff[1]*x**1 + coeff[2]*x**0,6)

# def df(x,coeff):
#     return round(5*coeff[0]*x**4 + 4*coeff[1]*x**3 +3*coeff[2]*x**2 + 2*coeff[3]*x**1 + coeff[4]*x**0,6)

def df(x,coeff):
    y=0
    j=len(coeff)
    for k in range(j-1):
        y += (j-k-1)*coeff[k]*x**(j-k-2)
    return round(y,6)

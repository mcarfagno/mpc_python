import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import cvxpy as cp

def get_linear_model(x_bar,u_bar):
    """
    """

    # Control problem statement.

    N = 5 #number of state variables
    M = 2 #number of control variables
    T = 20 #Prediction Horizon
    dt = 0.25 #discretization step

    x = x_bar[0]
    y = x_bar[1]
    theta = x_bar[2]
    psi = x_bar[3]
    cte = x_bar[4]
    
    v = u_bar[0]
    w = u_bar[1]
    
    A = np.zeros((N,N))
    A[0,2]=-v*np.sin(theta)
    A[1,2]=v*np.cos(theta)
    A[4,3]=v*np.cos(-psi)
    A_lin=np.eye(N)+dt*A
    
    B = np.zeros((N,M))
    B[0,0]=np.cos(theta)
    B[1,0]=np.sin(theta)
    B[2,1]=1
    B[3,1]=-1
    B[4,0]=np.sin(-psi)
    B_lin=dt*B
    
    f_xu=np.array([v*np.cos(theta),v*np.sin(theta),w,-w,v*np.sin(-psi)]).reshape(N,1)
    C_lin = dt*(f_xu - np.dot(A,x_bar.reshape(N,1)) - np.dot(B,u_bar.reshape(M,1)))
    
    return A_lin,B_lin,C_lin

def calc_err(state,path):
    """
    Finds psi and cte w.r.t. the closest waypoint.

    :param state: array_like, state of the vehicle [x_pos, y_pos, theta]
    :param path: array_like, reference path ((x1, x2, ...), (y1, y2, ...), (th1 ,th2, ...)]
    :returns: (float,float)
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

    path_ref_vect = [np.cos(path[2,target_idx] + np.pi / 2),
                     np.sin(path[2,target_idx] + np.pi / 2)]
    
    #heading error w.r.t path frame
    psi = path[2,target_idx] - state[2]
        
    # the cross-track error is given by the scalar projection of the car->wp vector onto the faxle versor
    #cte = np.dot([dx[target_idx], dy[target_idx]],front_axle_vect)
    cte = np.dot([dx[target_idx], dy[target_idx]],path_ref_vect)

    return target_idx,psi,cte

def optimize(starting_state,u_bar,track);
    '''
    :param starting_state:
    :param u_bar:
    :param track:
    :returns:
    '''

    MAX_SPEED = 1.25
    MIN_SPEED = 0.75
    MAX_STEER_SPEED = 1.57/2

    N = 5 #number of state variables
    M = 2 #number of control variables
    T = 20 #Prediction Horizon
    dt = 0.25 #discretization step
    
    #Starting Condition
    x0 = np.zeros(N)
    x0[0] = starting_state[0]
    x0[1] = starting_state[1]
    x0[2] = starting_state[2]
    _,psi,cte = calc_err(x0,track)
    x0[3]=psi
    x0[4]=cte

    # Prediction
    x_bar=np.zeros((N,T+1))
    x_bar[:,0]=x0

    for t in range (1,T+1):
        xt=x_bar[:,t-1].reshape(5,1)
        ut=u_bar[:,t-1].reshape(2,1)

        A,B,C=get_linear_model(xt,ut)

        xt_plus_one = np.squeeze(np.dot(A,xt)+np.dot(B,ut)+C)

        _,psi,cte = calc_err(xt_plus_one,track)
        xt_plus_one[3]=psi
        xt_plus_one[4]=cte

        x_bar[:,t]= xt_plus_one

    #CVXPY Linear MPC problem statement
    cost = 0
    constr = []
    x = cp.Variable((N, T+1))
    u = cp.Variable((M, T))

    for t in range(T):

        # Tracking
        if t > 0:
            idx,_,_ = calc_err(x_bar[:,t],track)
            delta_x = track[:,idx]-x[0:3,t]
            cost+= cp.quad_form(delta_x,10*np.eye(3))
            
        # Tracking last time step
        if t == T:
            idx,_,_ = calc_err(x_bar[:,t],track)
            delta_x = track[:,idx]-x[0:3,t]
            cost+= cp.quad_form(delta_x,100*np.eye(3))

        # Actuation rate of change
        if t < (T - 1):
            cost += cp.quad_form(u[:, t + 1] - u[:, t], 25*np.eye(M))
        
        # Actuation effort
        cost += cp.quad_form( u[:, t],1*np.eye(M))
        
        # Constrains
        A,B,C=get_linear_model(x_bar[:,t],u_bar[:,t])
        constr += [x[:,t+1] == A*x[:,t] + B*u[:,t] + C.flatten()]

    # sums problem objectives and concatenates constraints.
    constr += [x[:,0] == x_sim[:,sim_time]] # starting condition
    constr += [u[0, :] <= MAX_SPEED]
    constr += [u[0, :] >= MIN_SPEED]
    constr += [cp.abs(u[1, :]) <= MAX_STEER_SPEED]
    
    # Solve
    prob = cp.Problem(cp.Minimize(cost), constr)
    solution = prob.solve(solver=cp.ECOS, verbose=False)
    
    #retrieved optimized U and assign to u_bar to linearize in next step
    u_bar=np.vstack((np.array(u.value[0, :]).flatten(),
                    (np.array(u.value[1, :]).flatten())))
    
    return u_bar
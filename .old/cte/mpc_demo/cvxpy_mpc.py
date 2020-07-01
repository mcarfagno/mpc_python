import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.integrate import odeint
from scipy.interpolate import interp1d
import cvxpy as cp

from utils import road_curve, f, df

from mpc_config import Params
P=Params()

def get_linear_model(x_bar,u_bar):
    """
    """

    x = x_bar[0]
    y = x_bar[1]
    theta = x_bar[2]

    v = u_bar[0]
    w = u_bar[1]

    A = np.zeros((P.N,P.N))
    A[0,2]=-v*np.sin(theta)
    A[1,2]=v*np.cos(theta)
    A_lin=np.eye(P.N)+P.dt*A

    B = np.zeros((P.N,P.M))
    B[0,0]=np.cos(theta)
    B[1,0]=np.sin(theta)
    B[2,1]=1
    B_lin=P.dt*B

    f_xu=np.array([v*np.cos(theta),v*np.sin(theta),w]).reshape(P.N,1)
    C_lin = P.dt*(f_xu - np.dot(A,x_bar.reshape(P.N,1)) - np.dot(B,u_bar.reshape(P.M,1)))

    return A_lin,B_lin,C_lin


def optimize(state,u_bar,track):
    '''
    :param state:
    :param u_bar:
    :param track:
    :returns:
    '''

    MAX_SPEED = 1.25
    MIN_SPEED = 0.75
    MAX_STEER_SPEED = 1.57/2

    # compute polynomial coefficients of the track
    K=road_curve(state,track)

    # dynamics starting state w.r.t vehicle frame
    x_bar=np.zeros((P.N,P.T+1))

    #prediction for linearization of costrains
    for t in range (1,P.T+1):
        xt=x_bar[:,t-1].reshape(P.N,1)
        ut=u_bar[:,t-1].reshape(P.M,1)
        A,B,C=get_linear_model(xt,ut)
        xt_plus_one = np.squeeze(np.dot(A,xt)+np.dot(B,ut)+C)
        x_bar[:,t]= xt_plus_one

    #CVXPY Linear MPC problem statement
    cost = 0
    constr = []
    x = cp.Variable((P.N, P.T+1))
    u = cp.Variable((P.M, P.T))

    for t in range(P.T):

        #cost += 30*cp.sum_squares(x[2,t]-np.arctan(df(x_bar[0,t],K))) # psi
        cost += 50*cp.sum_squares(x[2,t]-np.arctan2(df(x_bar[0,t],K),x_bar[0,t])) # psi
        cost += 20*cp.sum_squares(f(x_bar[0,t],K)-x[1,t]) # cte

        # Actuation rate of change
        if t < (P.T - 1):
            cost += cp.quad_form(u[:, t + 1] - u[:, t], 100*np.eye(P.M))

        # Actuation effort
        cost += cp.quad_form( u[:, t],1*np.eye(P.M))

        # Kinrmatics Constrains (Linearized model)
        A,B,C=get_linear_model(x_bar[:,t],u_bar[:,t])
        constr += [x[:,t+1] == A@x[:,t] + B@u[:,t] + C.flatten()]

    # sums problem objectives and concatenates constraints.
    constr += [x[:,0] == x_bar[:,0]] #<--watch out the start condition
    constr += [u[0, :] <= MAX_SPEED]
    constr += [u[0, :] >= MIN_SPEED]
    constr += [cp.abs(u[1, :]) <= MAX_STEER_SPEED]

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constr)
    solution = prob.solve(solver=cp.OSQP, verbose=False)

    #retrieved optimized U and assign to u_bar to linearize in next step
    u_bar=np.vstack((np.array(u.value[0, :]).flatten(),
                    (np.array(u.value[1, :]).flatten())))

    return u_bar

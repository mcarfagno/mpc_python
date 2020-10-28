import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from scipy.integrate import odeint
from scipy.interpolate import interp1d
import cvxpy as cp

from utils import *

from mpc_config import Params
P=Params()

def get_linear_model(x_bar,u_bar):
    """
    """
    L=0.3

    x = x_bar[0]
    y = x_bar[1]
    v = x_bar[2]
    theta = x_bar[3]

    a = u_bar[0]
    delta = u_bar[1]

    A = np.zeros((P.N,P.N))
    A[0,2]=np.cos(theta)
    A[0,3]=-v*np.sin(theta)
    A[1,2]=np.sin(theta)
    A[1,3]=v*np.cos(theta)
    A[3,2]=v*np.tan(delta)/L
    A_lin=np.eye(P.N)+P.dt*A

    B = np.zeros((P.N,P.M))
    B[2,0]=1
    B[3,1]=v/(L*np.cos(delta)**2)
    B_lin=P.dt*B

    f_xu=np.array([v*np.cos(theta), v*np.sin(theta), a,v*np.tan(delta)/L]).reshape(P.N,1)
    C_lin = P.dt*(f_xu - np.dot(A,x_bar.reshape(P.N,1)) - np.dot(B,u_bar.reshape(P.M,1)))

    return np.round(A_lin,4), np.round(B_lin,4), np.round(C_lin,4)


def optimize(state,u_bar,track,ref_vel=1.):
    '''
    :param state:
    :param u_bar:
    :param track:
    :returns:
    '''

    MAX_SPEED = 1.5 #m/s
    MAX_ACC = 1.0 #m/ss
    MAX_D_ACC = 1.0 #m/sss
    MAX_STEER = np.radians(30) #rad
    MAX_D_STEER = np.radians(30) #rad/s

    REF_VEL = ref_vel #m/s
    
    # dynamics starting state
    x_bar = np.zeros((P.N,P.T+1))
    x_bar[:,0] = state

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
    
    # Cost Matrices
    Q = np.diag([20,20,10,0]) #state error cost
    Qf = np.diag([10,10,10,10]) #state final error cost
    R = np.diag([10,10])       #input cost
    R_ = np.diag([10,10])      #input rate of change cost

    #Get Reference_traj
    x_ref, d_ref = get_ref_trajectory(x_bar[:,0] ,track, REF_VEL)

    #Prediction Horizon
    for t in range(P.T):

        # Tracking Error
        cost += cp.quad_form(x[:,t] - x_ref[:,t], Q)

        # Actuation effort
        cost += cp.quad_form(u[:,t], R)

        # Actuation rate of change
        if t < (P.T - 1):
            cost += cp.quad_form(u[:,t+1] - u[:,t], R_)
            constr+= [cp.abs(u[0, t + 1] - u[0, t])/P.dt <= MAX_D_ACC]    #max acc rate of change
            constr += [cp.abs(u[1, t + 1] - u[1, t])/P.dt <= MAX_D_STEER] #max steer rate of change

        # Kinrmatics Constrains (Linearized model)
        A,B,C = get_linear_model(x_bar[:,t], u_bar[:,t])
        constr += [x[:,t+1] == A@x[:,t] + B@u[:,t] + C.flatten()]

    # sums problem objectives and concatenates constraints.
    constr += [x[:,0] == x_bar[:,0]]           #starting condition
    constr += [x[2,:] <= MAX_SPEED]           #max speed
    constr += [x[2,:] >= 0.0]                 #min_speed (not really needed)
    constr += [cp.abs(u[0,:]) <= MAX_ACC]     #max acc
    constr += [cp.abs(u[1,:]) <= MAX_STEER]   #max steer

    # Solve
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, verbose=False)

    if "optimal" not in prob.status:
        print("WARN: No optimal solution")
        return u_bar

    #retrieved optimized U and assign to u_bar to linearize in next step
    u_opt=np.vstack((np.array(u.value[0, :]).flatten(),
                    (np.array(u.value[1, :]).flatten())))

    return u_opt

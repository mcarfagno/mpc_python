#! /usr/bin/env python

import rospy
import numpy as np

from nav_msgs.msg import Odometry

from geometry_msgs.msg import Twist

from utils import compute_path_from_wp
from cvxpy_mpc import optimize

# classes
class Node():

    def __init__(self):

        rospy.init_node('mpc_node')

        N = 5 #number of state variables
        M = 2 #number of control variables
        T = 20 #Prediction Horizon
        dt = 0.25 #discretization step

        # State for the robot mathematical model
        self.state =  None

        # starting guess output
        self.opt_u =  np.zeros((M,T))
        self.opt_u[0,:] = 1 #m/s
        self.opt_u[1,:] = np.radians(0) #rad/s

        # Interpolated Path to follow given waypoints
        self.path = compute_path_from_wp([0,20,30,30],[0,0,10,20])

        self._cmd_pub = rospy.Publisher(rospy.get_namespace() + 'husky_velocity_controller/cmd_vel', Twist, queue_size=10)
        self._odom_sub = rospy.Subscriber(rospy.get_namespace() +'husky_velocity_controller/odom', Odometry, self._odom_cb, queue_size=1)

    def run(self):
        while 1:
            if self.state is not None:

                #optimization loop
                self.opt_u = optimize(self.state,
                                        self.opt_u,
                                        self.path)
                
                msg = Twist()
                msg.linear.x=self.opt_u[0,1]
                msg.angular.z=self.opt_u[0,1]

                self._cmd_pub(msg)

    def _odom_cb(self,odom):   
        '''
        Updates state with latest odometry.

        :param odom: nav_msgs.msg.Odometry
        '''

        state = np.zeros(3)

        # Update current position
        state[0] = odom.pose.pose.position.x
        state[1] = odom.pose.pose.position.y

        # Update current orientation
        _, _, state[2] = euler_from_quaternion(
                        [odom.pose.pose.orientation.x,
                         odom.pose.pose.orientation.y, 
                         odom.pose.pose.orientation.z, 
                         odom.pose.pose.orientation.w])
        
        self.state = state

def main():
    ros_node=Node()
    try:
        ros_node.run()
    except rospy.exceptions.ROSException as e:
        sys.exit(e)

if __name__ == '__main__':
    main()
import numpy as np


class VehicleModel:
    """
    Helper class that contains the parameters of the vehicle to be controlled

    Attributes:
        wheelbase: [m]
        max_speed: [m/s]
        max_acc: [m/ss]
        max_d_acc: [m/sss]
        max_steer: [rad]
        max_d_steer: [rad/s]
    """

    def __init__(self):
        self.wheelbase = 0.3
        self.max_speed = 1.5
        self.max_acc = 1.0
        self.max_d_acc = 1.0
        self.max_steer = np.radians(30)
        self.max_d_steer = np.radians(30)

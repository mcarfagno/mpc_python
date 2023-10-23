import numpy as np


class VehicleModel:
    """
    Helper class to hold vehicle info
    """

    def __init__(self):
        self.wheelbase = 0.3  # vehicle wheelbase [m]
        self.max_speed = 1.5  # [m/s]
        self.max_acc = 1.0  # [m/ss]
        self.max_d_acc = 1.0  # [m/sss]
        self.max_steer = np.radians(30)  # [rad]
        self.max_d_steer = np.radians(30)  # [rad/s]

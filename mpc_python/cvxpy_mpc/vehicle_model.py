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
        self.wheelbase = (
            0.2965  # front axle (0.1385) - rear axle (-0.158), from buddy.xml
        )
        self.max_speed = 1.5
        self.max_acc = 1.0
        self.max_jerk = 1.0
        # also often noted as delta and delta_dot
        self.max_steer = 0.38  # ±0.38 rad (±21.8°), from buddy.xml steering joint range
        self.max_steer_rate = 0.38

import numpy as np


class CubatureKalmanFilter(object):
    count = 0

    def __init__(self, dim_x, dim_z, points):
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        # self.P[4:, 4:] = 0.01
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = np.eye(dim_x) * 0.1
        self.R = np.eye(dim_z) * 0.1
        self._dim_x = dim_x
        self._dim_z = dim_z
        self._num_sigmas = points.num_sigmas()

        self.K = np.zeros((dim_x, dim_z))
        self.z = np.array([[0] * dim_z]).T
        self.H = np.zeros((dim_z, dim_x))

        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_x))

        self.inv = np.linalg.inv  # 求逆

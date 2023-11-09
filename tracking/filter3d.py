import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import cholesky

from tracking.CKF import CubatureKalmanFilter
from tracking.UKF import UnscentedKalmanFilter
from tracking.sigma import sigma, _sigma


class UKF(object):
    count = 0

    def __init__(self, bbox3D):
        self.dt = 0.1

        self.points = sigma(7, alpha=.1, beta=2., kappa=-1)
        self.ukf = UnscentedKalmanFilter(dim_x=7, dim_z=4, points=self.points)

        self.sqrt = cholesky
        self.residual_x = np.subtract
        self.residual_z = np.subtract

        self.num_sigma = 15

        self.ukf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0]])

        self.ukf.P[4:, 4:] *= 1000.
        self.ukf.P *= 10.
        self.ukf.G = np.array([[0.5 * np.cos(bbox3D[3]) * self.dt * self.dt, 0, 0],
                               [0.5 * np.sin(bbox3D[3]) * self.dt * self.dt, 0, 0],
                               [0, 0, 0.5 * self.dt * self.dt],
                               [0, 0.5 * self.dt * self.dt, 0],
                               [self.dt, 0, 0],
                               [0, self.dt, 0],
                               [0, 0, self.dt]])
        u = np.eye(3) * 0.01
        self.ukf.Q = np.dot(np.dot(self.ukf.G, u), self.ukf.G.T)
        self.ukf.x[0:4] = bbox3D[0:4]
        self.pose = bbox3D[4:7]

    def predict(self):
        sigmas = self.points.sigma_point(self.ukf.x, self.ukf.P)  # 2、生成sigma点
        """
        sigmas: (p_x, p_y, p_z, theta, w, h, l, v, theta_r, v_z)
        """

        """
        3、预测，生成sigma-
        """
        self.ukf.x_prior = np.zeros(7)
        self.ukf.P_prior = np.eye(7)  # 记得归零啊，血和泪的教训！！！
        for i, s in enumerate(sigmas):
            if s[5] != 0:
                self.ukf.sigmas_f[i][0] = s[0] + s[4] / s[5] * (np.sin(s[5] * self.dt + s[3]) - np.sin(s[3]))
                self.ukf.sigmas_f[i][1] = s[1] + s[4] / s[5] * (-np.cos(s[5] * self.dt + s[3]) + np.cos(s[3]))
                self.ukf.sigmas_f[i][3] = s[3] + s[5] * self.dt
            else:
                self.ukf.sigmas_f[i][0] = s[0] + s[4] * np.cos(s[3]) * self.dt
                self.ukf.sigmas_f[i][1] = s[1] + s[4] * np.sin(s[3]) * self.dt
                self.ukf.sigmas_f[i][3] = s[3]
            self.ukf.sigmas_f[i][2] = s[2] + s[6] * self.dt
            self.ukf.sigmas_f[i][4] = s[4]
            self.ukf.sigmas_f[i][5] = s[5]
            self.ukf.sigmas_f[i][6] = s[6]

        """
        4、权重更新先验估计值和P
        """
        for i, s in enumerate(self.ukf.sigmas_f):
            self.ukf.x_prior += self.points.Wm[i] * s
        for i, s in enumerate(self.ukf.sigmas_f):
            self.ukf.P_prior += self.points.Wc[i] * np.dot((self.ukf.x_prior - self.ukf.sigmas_f[i]).T,
                                                           (self.ukf.x_prior - self.ukf.sigmas_f[i]))
        self.ukf.P_prior = self.ukf.P_prior + self.ukf.Q
        pose = self.ukf.x_prior[:4]
        if self.ukf.x_prior[3] >= np.pi:
            self.ukf.x_prior[3] -= np.pi * 2
        if self.ukf.x_prior[3] < -np.pi:
            self.ukf.x_prior[3] += np.pi * 2
        return pose

    def update(self, z):
        """
        5、观测估计(pass)
        6、观测估计（pass）
        7、卡尔曼增益
        """
        if self.ukf.x_prior[3] >= np.pi:
            self.ukf.x_prior[3] -= np.pi * 2
        if self.ukf.x_prior[3] < -np.pi:
            self.ukf.x_prior[3] += np.pi * 2

        if z[3] >= np.pi:
            z[3] -= np.pi * 2
        if z[3] <= -np.pi:
            z[3] += np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > np.pi / 2.0 and abs(z[3] - self.ukf.x_prior[3]) < np.pi / 2.0 * 3:
            self.ukf.x_prior[3] += np.pi
            if self.ukf.x_prior[3] >= np.pi:
                self.ukf.x_prior[3] -= np.pi * 2
            if self.ukf.x_prior[3] < -np.pi:
                self.ukf.x_prior[3] += np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > np.pi / 2.0 * 3:
            if z[3] > 0:
                self.ukf.x_prior[3] += np.pi * 2
            else:
                self.ukf.x_prior[3] -= np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > np.pi / 2.0 and abs(z[3] - self.ukf.x_prior[3]) < np.pi / 2.0 * 3:
            z[3] -= np.pi
            if z[3] >= np.pi:
                z[3] -= np.pi * 2
            if z[3] <= -np.pi:
                z[3] += np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > np.pi / 2.0 * 3:
            if z[3] > 0:
                self.ukf.x_prior[3] += np.pi * 2
            else:
                self.ukf.x_prior[3] -= np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > 0.5:
            self.ukf.x_prior[5] = 0

        self.ukf.K = np.dot(np.dot(self.ukf.P_prior, self.ukf.H.T),
                            self.ukf.inv(np.dot(np.dot(self.ukf.H, self.ukf.P_prior), self.ukf.H.T) + self.ukf.R))

        """
        8、更新
        """
        self.ukf.x = self.ukf.x_prior + np.dot(self.ukf.K, (z[0:4] - np.dot(self.ukf.H, self.ukf.x_prior)))
        self.ukf.P = self.ukf.P_prior - np.dot(self.ukf.K, np.dot(self.ukf.H, self.ukf.P_prior))
        if self.ukf.x[3] >= np.pi:
            self.ukf.x[3] -= np.pi * 2
        if self.ukf.x[3] < -np.pi:
            self.ukf.x[3] += np.pi * 2
        self.pose = z[4:7]

    def unmatch_update(self):
        self.ukf.x = self.ukf.x_prior
        self.ukf.P = self.ukf.P_prior


class _UKF(object):
    count = 0

    def __init__(self, bbox3D):
        self.dt = 0.1

        self.points = sigma(6, alpha=.1, beta=2., kappa=-1)
        self.ukf = UnscentedKalmanFilter(dim_x=6, dim_z=4, points=self.points)

        self.sqrt = cholesky
        self.residual_x = np.subtract
        self.residual_z = np.subtract

        self.num_sigma = 13

        self.ukf.H = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0]])

        self.ukf.G = np.array([[0.5 * np.cos(bbox3D[3]) * self.dt * self.dt, 0, 0],
                               [0.5 * np.sin(bbox3D[3]) * self.dt * self.dt, 0, 0],
                               [0, 0, 0.5 * self.dt * self.dt],
                               [0, self.dt, 0],
                               [self.dt, 0, 0],
                               [0, 0, self.dt]])
        u = np.eye(3) * 0.01
        self.ukf.Q = np.dot(np.dot(self.ukf.G, u), self.ukf.G.T)
        self.ukf.x[0:4] = bbox3D[0:4]
        self.pose = bbox3D[4:]

    def predict(self):

        self.ukf.G = np.array([[0.5 * np.cos(self.ukf.x[3]) * self.dt * self.dt, 0, 0],
                               [0.5 * np.sin(self.ukf.x[3]) * self.dt * self.dt, 0, 0],
                               [0, 0, 0.5 * self.dt * self.dt],
                               [0, self.dt, 0],
                               [self.dt, 0, 0],
                               [0, 0, self.dt]])
        u = np.eye(3) * 0.01
        self.ukf.Q = np.dot(np.dot(self.ukf.G, u), self.ukf.G.T)

        sigmas = self.points.sigma_point(self.ukf.x, self.ukf.P)  # 2、生成sigma点
        """
        sigmas: (p_x, p_y, p_z, theta, w, h, l, v, theta_r, v_z)
        """

        """
        3、预测，生成sigma-
        """
        for i, s in enumerate(sigmas):
            self.ukf.sigmas_f[i][0] = s[0] + s[4] * np.cos(s[3]) * self.dt
            self.ukf.sigmas_f[i][1] = s[1] + s[4] * np.sin(s[3]) * self.dt
            self.ukf.sigmas_f[i][3] = s[3]
            self.ukf.sigmas_f[i][2] = s[2] + s[5] * self.dt
            self.ukf.sigmas_f[i][4] = s[4]
            self.ukf.sigmas_f[i][5] = s[5]

        """
        4、权重更新先验估计值和P
        """
        self.ukf.x_prior = np.zeros(6)
        self.ukf.P_prior = np.eye(6)  # 记得归零啊，血和泪的教训！！！
        for i, s in enumerate(self.ukf.sigmas_f):
            self.ukf.x_prior += self.points.Wm[i] * s
        for i, s in enumerate(self.ukf.sigmas_f):
            self.ukf.P_prior += self.points.Wm[i] * np.dot((self.ukf.x_prior - self.ukf.sigmas_f[i]).T,
                                                           (self.ukf.x_prior - self.ukf.sigmas_f[i]))
        self.ukf.P_prior = self.ukf.P_prior + self.ukf.Q
        pose = self.ukf.x_prior[:4]
        if self.ukf.x_prior[3] >= np.pi:
            self.ukf.x_prior[3] -= np.pi * 2
        if self.ukf.x_prior[3] < -np.pi:
            self.ukf.x_prior[3] += np.pi * 2
        return pose

    def update(self, z):
        """
        5、观测估计(pass)
        6、观测估计（pass）
        7、卡尔曼增益
        """
        if self.ukf.x_prior[3] >= np.pi:
            self.ukf.x_prior[3] -= np.pi * 2
        if self.ukf.x_prior[3] < -np.pi:
            self.ukf.x_prior[3] += np.pi * 2

        if z[3] >= np.pi:
            z[3] -= np.pi * 2
        if z[3] <= -np.pi:
            z[3] += np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > np.pi / 2.0 and abs(z[3] - self.ukf.x_prior[3]) < np.pi / 2.0 * 3:
            self.ukf.x_prior[3] += np.pi
            if self.ukf.x_prior[3] >= np.pi:
                self.ukf.x_prior[3] -= np.pi * 2
            if self.ukf.x_prior[3] < -np.pi:
                self.ukf.x_prior[3] += np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > np.pi / 2.0 * 3:
            if z[3] > 0:
                self.ukf.x_prior[3] += np.pi * 2
            else:
                self.ukf.x_prior[3] -= np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > np.pi / 2.0 and abs(z[3] - self.ukf.x_prior[3]) < np.pi / 2.0 * 3:
            z[3] -= np.pi
            if z[3] >= np.pi:
                z[3] -= np.pi * 2
            if z[3] <= -np.pi:
                z[3] += np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > np.pi / 2.0 * 3:
            if z[3] > 0:
                self.ukf.x_prior[3] += np.pi * 2
            else:
                self.ukf.x_prior[3] -= np.pi * 2

        if abs(z[3] - self.ukf.x_prior[3]) > 0.5:
            self.ukf.x_prior[5] = 0
        self.ukf.K = np.dot(np.dot(self.ukf.P_prior, self.ukf.H.T),
                            self.ukf.inv(np.dot(np.dot(self.ukf.H, self.ukf.P_prior), self.ukf.H.T) + self.ukf.R))

        """
        8、更新
        """
        self.ukf.x = self.ukf.x_prior + np.dot(self.ukf.K, (z[0:4] - np.dot(self.ukf.H, self.ukf.x_prior)))
        self.ukf.P = self.ukf.P_prior - np.dot(self.ukf.K, np.dot(self.ukf.H, self.ukf.P_prior))
        if self.ukf.x[3] >= np.pi:
            self.ukf.x[3] -= np.pi * 2
        if self.ukf.x[3] < -np.pi:
            self.ukf.x[3] += np.pi * 2
        self.pose = z[4:7]

    def unmatch_update(self):
        self.ukf.x = self.ukf.x_prior
        self.ukf.P = self.ukf.P_prior


class CKF(object):
    count = 0

    def __init__(self, bbox3D,u):
        self.dt = 1

        self.points = _sigma(7)
        self.ckf = CubatureKalmanFilter(dim_x=7, dim_z=4, points=self.points)

        self.sqrt = cholesky
        self.residual_x = np.subtract
        self.residual_z = np.subtract

        self.num_sigma = 14
        """
        self.ckf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1]])
        """
        self.ckf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0]])
        self.ckf.P[4:, 4:] *= 1000.
        self.ckf.P *= 10.
        self.ckf.G = np.array([[0.5 * np.cos(bbox3D[3]) * self.dt * self.dt, 0, 0],
                               [0, 0, 0.5 * self.dt * self.dt],
                               [0.5 * np.sin(bbox3D[3]) * self.dt * self.dt, 0, 0],
                               [0, 0.5 * self.dt * self.dt, 0],
                               [self.dt, 0, 0],
                               [0, self.dt, 0],
                               [0, 0, self.dt]])
        u = np.eye(3) * 0.01
        self.ckf.Q = np.dot(np.dot(self.ckf.G, u), self.ckf.G.T)
        self.ckf.x[0:4] = bbox3D[0:4]
        self.pose = bbox3D[4:7]

    def predict(self):
        self.ckf.G = np.array([[0.5 * np.cos(self.ckf.x[3]) * self.dt * self.dt, 0, 0],
                               [0, 0, 0.5 * self.dt * self.dt],
                               [0.5 * np.sin(self.ckf.x[3]) * self.dt * self.dt, 0, 0],
                               [0, 0.5 * self.dt * self.dt, 0],
                               [self.dt, 0, 0],
                               [0, self.dt, 0],
                               [0, 0, self.dt]])
        u = np.eye(3) * 0.01
        self.ckf.Q = np.dot(np.dot(self.ckf.G, u), self.ckf.G.T)

        sigmas = self.points.sigma_point(self.ckf.x, self.ckf.P)  # 2、生成sigma点
        """
        sigmas: (p_x, p_y, p_z, theta, w, h, l, v, theta_r, v_z)
        """

        """
        3、预测，生成sigma-
        """
        # self.x = self.ckf.x_prior
        self.ckf.x_prior = np.zeros(7)
        self.ckf.P_prior = np.eye(7)  # 记得归零啊，血和泪的教训！！！
        for i, s in enumerate(sigmas):
            if s[5] != 0:
                self.ckf.sigmas_f[i][0] = s[0] + s[4] / s[5] * (np.sin(s[5] * self.dt + s[3]) - np.sin(s[3]))
                self.ckf.sigmas_f[i][2] = s[2] + s[4] / s[5] * (-np.cos(s[5] * self.dt + s[3]) + np.cos(s[3]))
                self.ckf.sigmas_f[i][3] = s[3] + s[5] * self.dt
            else:
                self.ckf.sigmas_f[i][0] = s[0] + s[4] * np.cos(s[3]) * self.dt
                self.ckf.sigmas_f[i][2] = s[2] + s[4] * np.sin(s[3]) * self.dt
                self.ckf.sigmas_f[i][3] = s[3]
            self.ckf.sigmas_f[i][1] = s[1] + s[6] * self.dt
            self.ckf.sigmas_f[i][4] = s[4]
            self.ckf.sigmas_f[i][5] = s[5]
            self.ckf.sigmas_f[i][6] = s[6]

        """
        4、权重更新先验估计值和P
        """
        for i, s in enumerate(self.ckf.sigmas_f):
            self.ckf.x_prior += self.ckf.sigmas_f[i]
        self.ckf.x_prior = self.ckf.x_prior / 14
        for i, s in enumerate(self.ckf.sigmas_f):
            self.ckf.P_prior += np.dot(np.asarray([self.ckf.sigmas_f[i]]).T, np.asarray([self.ckf.sigmas_f[i]]))
        self.ckf.P_prior = self.ckf.P_prior / 14 - np.dot(np.asarray([self.ckf.x_prior]).T,
                                                          np.asarray([self.ckf.x_prior])) + self.ckf.Q
        if self.ckf.x_prior[3] >= np.pi:
            self.ckf.x_prior[3] -= np.pi * 2
        if self.ckf.x_prior[3] < -np.pi:
            self.ckf.x_prior[3] += np.pi * 2
        pose = self.ckf.x_prior[:4]

        return pose

    def update(self, z,u):
        """
        5、观测估计(pass)
        6、观测估计（pass）
        7、卡尔曼增益

        if self.ckf.x_prior[3] - self.x[3] == 0:
            z_prior = np.array((((self.ckf.x_prior[2] - self.x[2]) / self.dt) / np.cos(self.ckf.x_prior[3]), 0,
                                (self.ckf.x_prior[2] - self.x[2]) / self.dt))
        else:
            z_prior = np.array((((self.ckf.x_prior[0] - self.x[0]) * (self.ckf.x_prior[3] - self.x[3]) / self.dt) / (
                    np.sin((self.ckf.x_prior[3] - self.x[3]) / self.dt + self.x[3]) - np.sin(+self.x[3])),
                                (self.ckf.x_prior[3] - self.x[3]) / self.dt,
                                (self.ckf.x_prior[2] - self.x[2]) / self.dt))
        z = np.concatenate((z, z_prior))
        """
        if self.ckf.x_prior[3] >= np.pi:
            self.ckf.x_prior[3] -= np.pi * 2
        if self.ckf.x_prior[3] < -np.pi:
            self.ckf.x_prior[3] += np.pi * 2

        if z[3] >= np.pi:
            z[3] -= np.pi * 2
        if z[3] <= -np.pi:
            z[3] += np.pi * 2

        if abs(z[3] - self.ckf.x_prior[3]) > np.pi / 2.0 and abs(z[3] - self.ckf.x_prior[3]) < np.pi / 2.0 * 3:
            self.ckf.x_prior[3] += np.pi
            if self.ckf.x_prior[3] >= np.pi:
                self.ckf.x_prior[3] -= np.pi * 2
            if self.ckf.x_prior[3] < -np.pi:
                self.ckf.x_prior[3] += np.pi * 2

        if abs(z[3] - self.ckf.x_prior[3]) > np.pi / 2.0 * 3:
            if z[3] > 0:
                self.ckf.x_prior[3] += np.pi * 2
            else:
                self.ckf.x_prior[3] -= np.pi * 2

        if abs(z[3] - self.ckf.x_prior[3]) > np.pi / 2.0 and abs(z[3] - self.ckf.x_prior[3]) < np.pi / 2.0 * 3:
            z[3] -= np.pi
            if z[3] >= np.pi:
                z[3] -= np.pi * 2
            if z[3] <= -np.pi:
                z[3] += np.pi * 2

        if abs(z[3] - self.ckf.x_prior[3]) > np.pi / 2.0 * 3:
            if z[3] > 0:
                self.ckf.x_prior[3] += np.pi * 2
            else:
                self.ckf.x_prior[3] -= np.pi * 2
        """
        if abs(z[3] - self.ckf.x_prior[3]) > 0.5:
            self.ckf.x_prior[5] = 0
        """
        self.ckf.K = np.dot(np.dot(self.ckf.P_prior, self.ckf.H.T),
                            self.ckf.inv(np.dot(np.dot(self.ckf.H, self.ckf.P_prior), self.ckf.H.T) + self.ckf.R))

        """
        8、更新
        """

        self.ckf.x = self.ckf.x_prior + np.dot(self.ckf.K, (z[0:4] - np.dot(self.ckf.H, self.ckf.x_prior)))
        self.ckf.P = self.ckf.P_prior - np.dot(self.ckf.K, np.dot(self.ckf.H, self.ckf.P_prior))
        if self.ckf.x[3] >= np.pi:
            self.ckf.x[3] -= np.pi * 2
        if self.ckf.x[3] < -np.pi:
            self.ckf.x[3] += np.pi * 2
        self.pose = z[4:7]

    def unmatch_update(self):
        self.ckf.x = self.ckf.x_prior
        self.ckf.P = self.ckf.P_prior


class _CKF(object):
    count = 0

    def __init__(self, bbox3D, score):
        self.dt = 1

        self.points = _sigma(8)
        self.ckf = CubatureKalmanFilter(dim_x=8, dim_z=4, points=self.points)

        self.sqrt = cholesky
        self.residual_x = np.subtract
        self.residual_z = np.subtract

        self.num_sigma = 16
        """
        self.ckf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1]])
        """
        self.ckf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0]])
        self.ckf.P[4:, 4:] *= 1000.
        self.ckf.P *= 10.
        self.ckf.G = np.array([[0.5 * self.dt * self.dt, 0, 0, 0],
                               [0, 0.5 * self.dt * self.dt, 0, 0],
                               [0, 0, 0.5 * self.dt * self.dt, 0],
                               [0, 0, 0, 0.5 * self.dt * self.dt],
                               [self.dt, 0, 0, 0],
                               [0, self.dt, 0, 0],
                               [0, 0, self.dt, 0],
                               [0, 0, 0, self.dt]])
        u = np.eye(4) * 0.01
        self.ckf.Q = np.dot(np.dot(self.ckf.G, u), self.ckf.G.T)
        self.ckf.x[0:4] = bbox3D[0:4]
        self.pose = bbox3D[4:7]
        self.score = score

    def predict(self):
        """
        self.ckf.G = np.array([[0.5 * np.cos(self.ckf.x[3]) * self.dt * self.dt, 0, 0],
                               [0, 0, 0.5 * self.dt * self.dt],
                               [0.5 * np.sin(self.ckf.x[3]) * self.dt * self.dt, 0, 0],
                               [0, 0.5 * self.dt * self.dt, 0],
                               [self.dt, 0, 0],
                               [0, self.dt, 0],
                               [0, 0, self.dt]])
        u = np.eye(3) * 0.01
        self.ckf.Q = np.dot(np.dot(self.ckf.G, u), self.ckf.G.T)
        """
        sigmas = self.points.sigma_point(self.ckf.x, self.ckf.P)  # 2、生成sigma点
        """
        sigmas: (p_x, p_y, p_z, theta, w, h, l, v, theta_r, v_z)
        """

        """
        3、预测，生成sigma-
        """
        # self.x = self.ckf.x_prior
        self.ckf.x_prior = np.zeros(8)
        self.ckf.P_prior = np.eye(8)  # 记得归零啊，血和泪的教训！！！
        for i, s in enumerate(sigmas):
            self.ckf.sigmas_f[i][0] = s[0] + s[4] * self.dt
            self.ckf.sigmas_f[i][1] = s[1] + s[5] * self.dt
            self.ckf.sigmas_f[i][2] = s[2] + s[6] * self.dt
            self.ckf.sigmas_f[i][3] = s[3] + s[7] * self.dt
            self.ckf.sigmas_f[i][4] = s[4]
            self.ckf.sigmas_f[i][5] = s[5]
            self.ckf.sigmas_f[i][6] = s[6]
            self.ckf.sigmas_f[i][7] = s[7]

        """
        4、权重更新先验估计值和P
        """
        # beta = 1 / self.score
        for i, s in enumerate(self.ckf.sigmas_f):
            self.ckf.x_prior += self.ckf.sigmas_f[i]
        self.ckf.x_prior = self.ckf.x_prior / 16
        for i, s in enumerate(self.ckf.sigmas_f):
            self.ckf.P_prior += np.dot(np.asarray([self.ckf.sigmas_f[i]]).T, np.asarray([self.ckf.sigmas_f[i]]))
        self.ckf.P_prior = self.ckf.P_prior / 16 - np.dot(np.asarray([self.ckf.x_prior]).T,
                                                          np.asarray([self.ckf.x_prior])) + self.ckf.Q
        if self.ckf.x_prior[3] >= np.pi:
            self.ckf.x_prior[3] -= np.pi * 2
        if self.ckf.x_prior[3] < -np.pi:
            self.ckf.x_prior[3] += np.pi * 2
        pose = self.ckf.x_prior[:4]

        return pose

    def update(self, z, score):
        """
        5、观测估计(pass)
        6、观测估计（pass）
        7、卡尔曼增益

        if self.ckf.x_prior[3] - self.x[3] == 0:
            z_prior = np.array((((self.ckf.x_prior[2] - self.x[2]) / self.dt) / np.cos(self.ckf.x_prior[3]), 0,
                                (self.ckf.x_prior[2] - self.x[2]) / self.dt))
        else:
            z_prior = np.array((((self.ckf.x_prior[0] - self.x[0]) * (self.ckf.x_prior[3] - self.x[3]) / self.dt) / (
                    np.sin((self.ckf.x_prior[3] - self.x[3]) / self.dt + self.x[3]) - np.sin(+self.x[3])),
                                (self.ckf.x_prior[3] - self.x[3]) / self.dt,
                                (self.ckf.x_prior[2] - self.x[2]) / self.dt))
        z = np.concatenate((z, z_prior))
        """
        # alpha = 1 / score

        if self.ckf.x_prior[3] >= np.pi:
            self.ckf.x_prior[3] -= np.pi * 2
        if self.ckf.x_prior[3] < -np.pi:
            self.ckf.x_prior[3] += np.pi * 2

        if z[3] >= np.pi:
            z[3] -= np.pi * 2
        if z[3] <= -np.pi:
            z[3] += np.pi * 2

        if abs(z[3] - self.ckf.x_prior[3]) > np.pi / 2.0 and abs(z[3] - self.ckf.x_prior[3]) < np.pi / 2.0 * 3:
            self.ckf.x_prior[3] += np.pi
            if self.ckf.x_prior[3] >= np.pi:
                self.ckf.x_prior[3] -= np.pi * 2
            if self.ckf.x_prior[3] < -np.pi:
                self.ckf.x_prior[3] += np.pi * 2

        if abs(z[3] - self.ckf.x_prior[3]) > np.pi / 2.0 * 3:
            if z[3] > 0:
                self.ckf.x_prior[3] += np.pi * 2
            else:
                self.ckf.x_prior[3] -= np.pi * 2
        """
        if abs(z[3] - self.ckf.x_prior[3]) > 0.5:
            if score > self.score:
                self.ckf.P_prior[3, 3] *= 1000.
            else:
                self.ckf.R[3, 3] *= 1000.
        
        if abs(z[3] - self.ckf.x_prior[3]) > np.pi / 2.0 and abs(z[3] - self.ckf.x_prior[3]) < np.pi / 2.0 * 3:
            z[3] -= np.pi
            if z[3] >= np.pi:
                z[3] -= np.pi * 2
            if z[3] <= -np.pi:
                z[3] += np.pi * 2
        
        if abs(z[3] - self.ckf.x_prior[3]) > np.pi / 2.0 * 3:
            if z[3] > 0:
                self.ckf.x_prior[3] += np.pi * 2
            else:
                self.ckf.x_prior[3] -= np.pi * 2
        """

        self.ckf.K = np.dot(np.dot(self.ckf.P_prior, self.ckf.H.T),
                            self.ckf.inv(
                                np.dot(np.dot(self.ckf.H, self.ckf.P_prior), self.ckf.H.T) + self.ckf.R))

        """
        8、更新
        """

        self.ckf.x = self.ckf.x_prior + np.dot(self.ckf.K, (z[0:4] - np.dot(self.ckf.H, self.ckf.x_prior)))
        self.ckf.P = self.ckf.P_prior - np.dot(self.ckf.K, np.dot(self.ckf.H, self.ckf.P_prior))
        if self.ckf.x[3] >= np.pi:
            self.ckf.x[3] -= np.pi * 2
        if self.ckf.x[3] < -np.pi:
            self.ckf.x[3] += np.pi * 2
        self.pose = z[4:7]
        self.score = 1 - ((1 - self.score) * (1 - score)) / ((1 - self.score) + (1 - score))

    def unmatch_update(self):
        self.ckf.x = self.ckf.x_prior
        self.ckf.P = self.ckf.P_prior


class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox3D, score):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],  # state transition matrix
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[0:, 0:] *= 0.1
        self.kf.P[4:, 4:] *= 10.
        self.kf.P *= 10.
        self.kf.G = np.array([[0.5, 0, 0, 0],
                              [0, 0.5, 0, 0],
                              [0, 0, 0.5, 0],
                              [0, 0, 0, 0.5],
                              [1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        u = np.eye(4) * 0.01
        self.kf.Q = np.dot(np.dot(self.kf.G, u), self.kf.G.T)
        self.history = []
        self.still_first = True
        self.kf.x[:4] = bbox3D.reshape((7, 1))[:4, :]  # [x,y,z,theta,l,w,h]
        self.pose = bbox3D[4:7]
        self.score = score

    def update(self, bbox3D, score):
        """
        Updates the state vector with observed bbox.
        """
        self.history = []
        # if self.still_first:
        #     self.first_continuing_hit += 1  # number of continuing hit in the fist time
        # ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        #########################     # flip

        if score != 0:
            alpha = 1 / score
            self.kf.R *= alpha
        else:
            self.kf.R *= 1000.

        self.kf.update(bbox3D[0:4])
        self.kf.R = np.eye(4) * 0.1

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.pose = bbox3D[4:7]
        # self.score = 1 - (1-self.score)*(1-score)  # 79.435 77.408  81.814  89.987  8
        # self.score = 1 - (1 - self.score) * (1 - score)/((1-self.score)+(1-score))  # 79.442    77.414  81.82   89.999  8
        self.score = (self.score**2+score**2) / (self.score + score)  # 79.446  77.415  81.829  90.011  7
        # self.score = max(self.score, score)  #79.441 77.406 81.828 89.987 7
        # self.score = np.sqrt(self.score**2+score**2) / (self.score + score)   # 79.442    77.415  81.82   89.999  8

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        if self.score != 0:
            beta = 1 / self.score
            self.kf.Q *= beta
        else:
            self.kf.Q *= 1000.

        self.kf.predict()
        u = np.eye(4) * 0.01
        self.kf.Q = np.dot(np.dot(self.kf.G, u), self.kf.G.T)
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.history.append(self.kf.x)
        pose = self.history[-1].tolist()
        pose = np.concatenate(pose[:4], axis=0)
        # pose = [pose[0][0],pose[1][0],pose[2][0],pose[3][0],pose[4][0],pose[5][0],pose[6][0]]
        # return self.history[-1]
        return pose

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:4].reshape((4,))

    def unmatch_update(self):
        self.kf.x = self.kf.x_prior
        self.kf.P = self.kf.P_prior
        self.score *= 0.8
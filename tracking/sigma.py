import numpy as np
from scipy.linalg import cholesky


class sigma(object):
    def __init__(self, n, alpha, beta, kappa):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.sqrt = cholesky
        self.subtract = np.subtract
        self._compute_weights()

    def sigma_point(self, x, P):
        x = np.asarray([x])
        P = np.eye(self.n) * P
        lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n
        U = self.sqrt((lambda_ + self.n) * P)

        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x
        for k in range(self.n):
            sigmas[k + 1] = self.subtract(x, U[k])
            sigmas[self.n + k + 1] = self.subtract(x, -U[k])
        return sigmas

    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.

        """

        n = self.n
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2 * n + 1, c)
        self.Wm = np.full(2 * n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)

    def num_sigmas(self):
        return 2 * self.n + 1


class _sigma(object):
    def __init__(self, n):
        self.n = n
        self.sqrt = cholesky
        self.subtract = np.subtract
        self.e = np.concatenate((np.eye(n), -np.eye(n)), axis=1)

    def sigma_point(self, x, P):
        x = np.asarray([x])
        a = self.sqrt(P)
        sigmas = np.zeros((self.n * 2, self.n))
        for i in range(2 * self.n):
            sigmas[i] = np.dot(a, self.e[:, i] * np.sqrt(self.n)) + x
        return sigmas

    def num_sigmas(self):
        return 2 * self.n

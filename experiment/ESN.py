import numpy as np


class EchoState:

    def __init__(self, theta_dim, u_dim, connectivity, spectral_radius, bias):
        """
        Initialize deep ESN W_in, W(s), W_trans(s) matrices
        :param theta_dim: used to initialize self.theta_dim
        :param u_dim: used to initialize self.u_dim
        :param connectivity: connectivity of the G matrices
        :param spectral_radius: spectral_radius of the G matrices

        :return self.G: reservoir transition matrix
        :return self.G_in: the input matrix
        :return self.b: the bias of transition
        """

        # State dimension
        self.theta_dim = theta_dim

        # Input dimension
        self.u_dim = u_dim

        # making reservoir transition matrix
        nans = np.random.randint(0, int(1 / connectivity) - 1, size=(theta_dim, theta_dim))
        G = np.random.uniform(-10, 10, theta_dim * theta_dim).reshape([theta_dim, theta_dim])
        G = np.where(nans, np.nan, G)
        G = np.nan_to_num(G)
        E, _ = np.linalg.eig(G)
        e_max = np.max(np.abs(E))
        G /= np.abs(e_max) / spectral_radius
        self.G = G

        # Making input matrix self.G_in
        self.G_in = np.random.uniform(-1, 1, self.theta_dim * self.u_dim).reshape([self.theta_dim, self.u_dim])
        u, s, vh = np.linalg.svd(self.G_in)
        scale = s[0]
        self.G_in = self.G_in / (scale * 1.2)

        # Attach bias
        self.b = np.ones(theta_dim) * bias + np.random.multivariate_normal(np.zeros(theta_dim), np.eye(theta_dim))

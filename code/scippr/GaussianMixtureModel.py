import numpy as np

class GaussianMixtureModel(object):

    def __init__(self):
        """
        A multidimensional mixture of Gaussians
        """
        return

    def set(self, pi, mu, Sigma):
        """
        Sets the parameter values for a multivariate Gaussian mixture model

        Parameters
        ----------
        pi: array, float
            array of length n_components containing coefficients on Gaussian
            components
        mu: array, float
            array of dimensions (n_components, n_variables) containing
            n_components means of Gaussians in n_variables dimensions
        Sigma: array, float
            array of dimensions (n_components, n_variables, n_variables)
            containing n_components covariance matrices of Gaussians in
            n_variables dimensions
        """
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma

        self.n_components = np.shape(self.pi)[0]
        self.n_variables = np.shape(self.mu)[1]
        return

    def evaluate_one(self, x):
        """
        Evaluates the Gaussian mixture probability at one point

        Parameters
        ----------
        x: array, float
            array of length n_variables at which to evaluate the Gaussian
            mixture model

        Returns
        -------
        p: float
            probability at x
        """
        p = 0.
        for k in range(self.n_components):
            p += self.pi[k]
                    / (2 * np.pi * np.sqrt(np.linalg.det(self.Sigma[k])))
                    * np.exp(-1. / 2. * (x - self.mu[k])
                    * np.linalg.inverse(self.Sigma[k]) * (x - self.mu[k]))
        return p

    def evaluate_ensemble(self, xs):
        """
        Evaluates the Gaussian mixture probability at an ensemble of points

        Parameters
        ----------
        xs: array, float
            array of dimensions (n_items, n_variables) at which to evaluate the
            Gaussian mixture model

        Returns
        -------
        p: float
            probability of xs
        """
        p = 1.
        for x in xs:
            p *= self.evaluate_one(x)
        return p

    def fit(self, xs, n_components):
        """
        Fits a multivariate Gaussian mixture model to data

        Parameters
        ----------
        xs: array, float
            array of dimensions (n_items, n_variables) to which the Gaussian
            mixture model will be Fits
        n_components: int
            number of components to consider
        """
        return

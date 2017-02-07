import scippr
from scippr import *

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import emcee

class LuminosityFunction(object):

    def __init__(self, data, SelectionFunction, Priors):
        """
        Redshift-dependent luminosity function object

        Parameters
        ----------
        data: array, float
            array of (n_items, 2) dimensions containing L_obs, z_obs values for
            all observed quasars
        SelectionFunction: [some kind of distribution function object]
            marginal probability distribution over all possible data values
            Parameters: theta
            Returns: probability
        Priors: [some kind of distribution function object]
            probability distribution over luminosity function parameters
            Parameters: theta (but defined by mu_0, A)
            Returns: prior probability
        """
        self.data = data
        self.SelectionFunction = SelectionFunction
        self.Priors = Priors
        self.n = np.shape(self.data)[0]
        self.chain = None

    def MarginalPosteriorForN(self, N, theta):
        """
        Evaluates the marginal posterior for the number of quasars given the
        distribution parameters and observed number of quasars

        Parameters
        ----------
        N: int
            total number of quasars at which to calculate probability
        theta: array, float
            parameter values at which to evaluate the marginal posterior for the
            total number of quasars

        Returns
        -------
        p: float
            probability associated with input parameter values
        """
        p = sp.stats.nbinom(self.n, 1. - self.SelectionFunction(theta)).pmf(N)
        return p

    def MarginalPosteriorForTheta(self, theta):
        """
        Evaluates the marginal posterior for the parameters given the data

        Parameters
        ----------
        theta: array, float
            parameter values at which to evaluate the marginal posterior for
            theta

        Returns
        -------
        p: float
            probability associated with input parameter values
        """
        p = self.Priors(theta) * self.SelectionFunction(theta)
            * GaussianMixtureModel(theta).evaluate_ensemble(self.data)
        return p

    def CompleteJointPosterior(self, theta):
        """
        Evaluates the complete joint posterior

        Parameters
        ----------
        theta: array, float
            parameter values at which to evaluate the complete joint posterior

        Returns
        -------
        p: float
            probability associated with input parameter values
        """
        p = MarginalPosteriorForN(theta[0], theta[1:])
            * MarginalPosteriorForTheta(theta[1:])
        return p

    def MCMC(self, initial_values, n_samples):
        """
        Conducts MCMC sampling of the CompleteJointPosterior

        Parameters
        ----------
        initial_values: array, float
            array of (n_walkers, n_params) dimensions containing initial values
            for the sampler
        n_samples: int
            number of samples to accept before terminating sampling

        Returns
        -------
        self.chain: array, float
            array of (n_walkers, n_samples, n_params) dimensions containing
            sampled parameter values, where parameters includes total number of
            quasars
        """
        (self.n_walkers, self.n_params) = np.shape(initial_values)
        self.sampler = emcee.EnsembleSampler(self.n_walkers, self.n_params,
                        self.CompleteJointPosterior)
        self.chain = self.sample(initial_values, n_samples)['chains']
        return self.chain

    def PlotLAndZ(self):
        """
        Plots the observed luminosity-redshift relation and the best-fit, as
        well as the marginal distributions in luminosity and redshift in both
        cases
        """
        if self.chain is not None:
            figure = ...
            figure.savefig('LAndZ.png')
        return

    def PlotPosterior(self):
        """
        Plots the best-fit luminosity function
        """
        if self.chain is not None:
            figure = ...
            figure.savefig('Posterior.png')
        return

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on October 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
import numpy as np
from scipy.cluster.vq import kmeans2

from sknetwork.utils.base import Algorithm


class KMeansDense(Algorithm):
    """Standard KMeansDense clustering based on SciPy function ``kmeans2``.

    Parameters
    ----------
    n_clusters :
        Number of desired clusters.
    init :
        Method for initialization. Available methods are ‘random’, ‘points’, ‘++’ and ‘matrix’:
        * ‘random’: generate k centroids from a Gaussian with mean and variance estimated from the data.
        * ‘points’: choose k observations (rows) at random from data for the initial centroids.
        * ‘++’: choose k observations accordingly to the kmeans++ method (careful seeding)
        * ‘matrix’: interpret the k parameter as a k by M (or length k array for one-dimensional data) array of initial
        centroids.
    n_init :
        Number of iterations of the k-means algorithm to run.
    tol :
        Relative tolerance with regards to inertia to declare convergence.

    Attributes
    ----------
    labels_ :
        Label of each sample.
    cluster_centers_ :
        A ‘k’ by ‘N’ array of centroids found at the last iteration of k-means.

    References
    ----------
    * MacQueen, J. (1967, June). Some methods for classification and analysis of multivariate observations.
      In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability
      (Vol. 1, No. 14, pp. 281-297).

    * Arthur, D., & Vassilvitskii, S. (2007, January). k-means++: The advantages of careful seeding.
      In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms (pp. 1027-1035).
      Society for Industrial and Applied Mathematics.
    """
    def __init__(self, n_clusters: int = 8, init: str = '++', n_init: int = 10, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.init = init.lower()
        self.n_init = n_init
        self.tol = tol

        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, x: np.ndarray) -> 'KMeansDense':
        """Fit algorithm to the data.

        Parameters
        ----------
        x:
            Data to cluster.

        Returns
        -------
        self: :class:`KMeansDense`
        """
        centroids, labels = kmeans2(data=x, k=self.n_clusters, iter=self.n_init, thresh=self.tol, minit=self.init)
        self.cluster_centers_ = centroids
        self.labels_ = labels

        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit algorithm to the data and return the labels.

        Parameters
        ----------
        x:
            Data to cluster.

        Returns
        -------
        labels: np.ndarray
        """
        self.fit(x)
        return self.labels_

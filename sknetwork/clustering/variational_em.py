#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2020
@author: Clément Bonet <cbonet@enst.fr>
"""
from copy import deepcopy
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD
from sknetwork.linalg import normalize


eps = np.finfo(float).eps
xmin = np.finfo(np.float).min
xmax = np.finfo(np.float).max


def likelihood(adjacency, taus, alphas, pis, n_clusters: int) -> float:
    """Compute the approximated likelihood

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (np.ndarray because of numba, could probably be modified in csr_matrix).
    taus:
        Taus matrix
    alphas:
        Alphas array
    pis:
        Pis matrix
    n_clusters:
        Number of clusters

    Returns
    -------
    likelihood: float
    """
    n = adjacency.shape[0]

    output = np.sum(taus.dot(np.log(alphas)))
    cpt = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                for q in range(n_clusters):
                    for r in range(n_clusters):
                        logb = (adjacency[i, j] * np.log(pis[q, r]) + (1-adjacency[i, j]) * np.log(1-pis[q, r]))
                        cpt += taus[i, q] * taus[j, r] * logb

    return output + cpt / 2 - np.sum(taus * np.log(taus))


def variational_step(adjacency, taus, alphas, pis, n_clusters: int):
    """Apply the variational step:
    - update membership_probas

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (np.ndarray because of numba, could probably be modified in csr_matrix).
    taus:
        Taus matrix
    alphas:
        Alphas array
    pis:
        Pis matrix
    n_clusters:
        Number of clusters

    Returns
    -------
    membership_probas:
        Taus array updated
    """
    n = adjacency.shape[0]
    log_tau = np.log(np.maximum(taus, eps))
    for i in range(n):
        log_tau[i, :] = np.log(alphas)
        for q in range(n_clusters):
            for j in range(n):
                if j != i:
                    for r in range(n_clusters):
                        log_tau[i, q] += taus[j, r] * (adjacency[i, j] * np.log(pis[q, r])
                                                       + (1-adjacency[i, j]) * np.log(1-pis[q, r]))

    log_tau = np.maximum(np.minimum(log_tau, xmax), xmin)
    tau = np.exp(log_tau)

    tau = normalize(tau, p=1)

    return np.maximum(tau, eps)


def maximization_step(adjacency, taus, pis, n_clusters: int):
    """Apply the maximization step:
    - update in place pis
    - update cluster_mean_probas

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (np.ndarray because of numba, could probably be modified in csr_matrix).
    taus:
        Taus matrix
    pis:
        Pis matrix
    n_clusters:
        Number of clusters

    Returns
    -------
    cluster_mean_probas:
        Alphas array updated
    """
    n = adjacency.shape[0]
    alphas = np.maximum(np.sum(taus, axis=0)/n, eps)

    for q in range(n_clusters):
        for r in range(n_clusters):
            num = 0
            denom = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        num += taus[i, q] * taus[j, r] * adjacency[i, j]
                        denom += taus[i, q] * taus[j, r]
            if denom > eps:
                pi = num / denom
            else:
                # class with a single vertex
                pi = 0.5

            pis[q, r] = np.minimum(np.maximum(pi, eps), 1-eps)

    return alphas


class VariationalEM(BaseClustering):
    """ Variational Expectation Maximization algorithm.

    Parameters
    ----------
    n_clusters:
        Number of desired clusters.
    init:
        {‘kmeans’, ‘random’}, defaults to ‘kmeans’
    max_iter:
        Maximum number of iterations to perform.
    tol:
        VEM will stop when the likelihood is not improved by more than tol, defaults 1e-6

    Attributes
    ----------
    labels_: np.ndarray
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix.
    adjacency_ : sparse.csr_matrix
        Adjacency matrix between clusters.

    Example
    -------
    >>> from sknetwork.clustering import VariationalEM
    >>> from sknetwork.data import karate_club
    >>> vem = VariationalEM(n_clusters=3)
    >>> adjacency = karate_club()
    >>> labels = vem.fit_transform(adjacency)
    >>> len(set(labels))
    3

    References
    ----------
    * Daudin, likelihood-likelihood., Picard, F., & Robin, S. (2008).
      `A mixture model for random graphs.
      <http://pbil.univ-lyon1.fr/members/fpicard/franckpicard_fichiers/pdf/DPR08.pdf>`_
      Statistics and computing, 2008

    * Miele, V., Picard, F., Daudin, likelihood-likelihood., Mariadassou, M. & Robin, S. (2007).
      `Technical documentation about estimation in the ERMG model.
      <http://www.math-evry.cnrs.fr/_media/logiciels/mixnet/mixnet-doc.pdf>`_
    """
    def __init__(self, n_clusters: int = 3, init: str = "kmeans",
                 max_iter: int = 100, tol: float = 1e-6, sort_clusters: bool = True,
                 return_membership: bool = True, return_aggregate: bool = True):
        super(VariationalEM, self).__init__(sort_clusters=sort_clusters,
                                            return_membership=return_membership,
                                            return_aggregate=return_aggregate)
        self.n_clusters = n_clusters
        if init != "kmeans" and init != "random":
            raise ValueError(
                "Unknown initialization. It must be either 'kmeans' or 'random'")
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

        self.cluster_mean_probas = None
        self.membership_probas = None
        self.pis = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'VariationalEM':
        """Apply the variational Expectation Maximization algorithm

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`VariationalEM`
        """
        adjacency = deepcopy(adjacency)

        # unweighted graphs
        adjacency[adjacency > 0] = 1
        adjacency[adjacency < 0] = 1

        self.cluster_mean_probas = np.ones(self.n_clusters) / self.n_clusters
        self.pis = np.zeros((self.n_clusters, self.n_clusters))
        n = adjacency.shape[0]

        if self.init == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, embedding_method=GSVD(self.n_clusters))
            labels = kmeans.fit_transform(adjacency)

            self.membership_probas = np.zeros(shape=(n, self.n_clusters))
            self.membership_probas[:] = np.eye(self.n_clusters)[labels]
        else:
            self.membership_probas = normalize(np.random.rand(n, self.n_clusters), p=1)

        likelihoods = []

        for k in range(self.max_iter):
            self.cluster_mean_probas = maximization_step(adjacency, self.membership_probas, self.pis, self.n_clusters)
            self.membership_probas = variational_step(adjacency, self.membership_probas, self.cluster_mean_probas,
                                                      self.pis, self.n_clusters)

            likelihoods.append(likelihood(adjacency, self.membership_probas, self.cluster_mean_probas, self.pis,
                                          self.n_clusters))

            if len(likelihoods) > 1 and (likelihoods[-1]-likelihoods[-2]) < self.tol:
                break

        self.labels_ = np.argmax(self.membership_probas, axis=1)
        self._secondary_outputs(adjacency)

        return self

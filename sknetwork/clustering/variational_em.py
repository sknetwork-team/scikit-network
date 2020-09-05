#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2020
@author: Clément Bonet <cbonet@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

import numba

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD

from copy import deepcopy


eps = np.finfo(float).eps
xmin = np.finfo(np.float).min
xmax = np.finfo(np.float).max


@numba.jit(nopython=True, parallel=True)
def J(adjacency, taus, alphas, pis, Q):
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
    Q:
        Number of clusters

    Returns
    -------
    J:
        J
    """
    n = adjacency.shape[0]

    output = np.sum(taus@np.log(alphas))

    cpt = 0
    for i in numba.prange(n):
        for j in numba.prange(n):
            if i != j:
                for q in numba.prange(Q):
                    for l in numba.prange(Q):
                        logb = (adjacency[i, j]*np.log(pis[q, l]) +
                                (1-adjacency[i, j])*np.log(1-pis[q, l]))
                        cpt += taus[i, q]*taus[j, l]*logb

    return output+cpt/2-np.sum(taus*np.log(taus))


@numba.jit(nopython=True, parallel=True)
def variational_step(adjacency, taus, alphas, pis, Q):
    """Apply the variational step:
    - update taus

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
    Q:
        Number of clusters

    Returns
    -------
    taus:
        Taus array updated
    """
    n = adjacency.shape[0]
    logTau = np.log(np.maximum(taus, eps))
    for i in numba.prange(n):
        logTau[i, :] = np.log(alphas)
        for q in numba.prange(Q):
            for j in numba.prange(n):
                if j != i:
                    for l in numba.prange(Q):
                        logTau[i, q] += taus[j, l]*(adjacency[i, j]*np.log(pis[q, l]) +
                                                    (1-adjacency[i, j])*np.log(1-pis[q, l]))

    logTau = np.maximum(np.minimum(logTau, xmax), xmin)
    tau = np.exp(logTau)

    for i in numba.prange(n):
        tau[i, :] /= np.sum(tau[i, :])

    return np.maximum(tau, eps)


@numba.jit(nopython=True, parallel=True)
def maximization_step(adjacency, taus, alphas, pis, Q):
    """Apply the maximization step:
    - update in place pis
    - update alphas

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
    Q:
        Number of clusters

    Returns
    -------
    alphas:
        Alphas array updated
    """
    n = adjacency.shape[0]
    alphas = np.maximum(np.sum(taus, axis=0)/n, eps)

    for q in numba.prange(Q):
        for l in range(Q):
            num = 0
            denom = 0
            for i in numba.prange(n):
                for j in numba.prange(n):
                    if i != j:
                        num += taus[i, q]*taus[j, l]*adjacency[i, j]
                        denom += taus[i, q]*taus[j, l]
            if denom > eps:
                pi = num/denom
            else:
                # class with a single vertex
                pi = 0.5

            pis[q, l] = np.minimum(np.maximum(pi, eps), 1-eps)

    return alphas


class VariationalEM(BaseClustering):
    """ Variational EM

    Parameters
    ----------
    n_clusters:
        Number of desired clusters.
    init:
        {‘kmeans’, ‘random’}, defaults to ‘kmeans’
    max_iter:
        Maximum number of iterations to perform.
    tol:
        VEM will stop when the likelihood is not improved more than tol, defaults 1e-6
    Attributes
    ----------
    labels: np.ndarray
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
    * Daudin, J-J., Picard, F., & Robin, S. (2008).
      `A mixture model for random graphs.
      <http://pbil.univ-lyon1.fr/members/fpicard/franckpicard_fichiers/pdf/DPR08.pdf>`_
      Statistics and computing, 2008

    * Miele, V., Picard, F., Daudin, J-J., Mariadassou, M. & Robin, S. (2007).
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

    def _init_vem(self, init):
        """Initialize the algorithm

        Parameters
        ----------
        init: str "kmeans" or "random"

        If init=="kmeans", initialize tau by the labels of kmeans
        else initialize randomly taus
        """
        n = self.adjacency.shape[0]

        if init == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters,
                            embedding_method=GSVD(self.n_clusters))
            labels_k = kmeans.fit_transform(self.adjacency)

            self.taus = np.zeros(shape=(n, self.n_clusters))
            self.taus[:] = np.eye(self.n_clusters)[labels_k]
        else:
            self.taus = np.random.rand(n, self.n_clusters)
            for i in range(n):
                self.taus[i, :] /= np.sum(self.taus[i, :])

        self.alphas = np.ones(shape=(self.n_clusters,))/self.n_clusters
        self.pis = np.zeros(shape=(self.n_clusters, self.n_clusters))

    def _j(self):
        """Compute the lower bound of log(p(X))
        (estimated likelihood)
        """
        n = self.adjacency.shape[0]
        return J(self.adjacency@np.eye(n), self.taus, self.alphas, self.pis, self.n_clusters)

    def _v_step(self):
        """Apply the variational step
        (ie update parameters taus)
        """
        n = self.adjacency.shape[0]
        self.taus = variational_step(self.adjacency @ np.eye(n), self.taus,
                                     self.alphas, self.pis, self.n_clusters)

    def _m_step(self):
        """Apply the maximization step
        (ie update parameters alphas and pis)
        """
        n = self.adjacency.shape[0]
        self.alphas = maximization_step(
            self.adjacency@np.eye(n), self.taus, self.alphas, self.pis, self.n_clusters)

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray]) -> 'VariationalEM':
        """Apply the variational EM algorithm

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.

        Returns
        -------
        self: :class:`VariationalEM`
        """
        self.adjacency = deepcopy(adjacency)

        ## unweighted graphs
        self.adjacency[self.adjacency > 0] = 1
        self.adjacency[self.adjacency < 0] = 1

        self._init_vem(self.init)

        Js = []

        for k in range(self.max_iter):
            self._m_step()
            self._v_step()

            Js.append(self._j())

            if len(Js) > 1 and Js[-1]-Js[-2] < self.tol:
                break

        self.labels_ = np.argmax(self.taus, axis=1)
        self._secondary_outputs(adjacency)

        return self

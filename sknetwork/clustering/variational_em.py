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


def likelihood(adjacency, membership_probs, cluster_mean_probs, cluster_transition_probs, n_clusters: int) -> float:
    """Compute the approximated likelihood

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (np.ndarray because of numba, could probably be modified in csr_matrix).
    membership_probs:
        Membership matrix given as a probability over clusters.
    cluster_mean_probs:
        Average value of cluster probability over nodes.
    cluster_transition_probs:
        Probabilities of transition from one cluster to another in one hop.
    n_clusters:
        Number of clusters

    Returns
    -------
    likelihood: float
    """
    n = adjacency.shape[0]

    output = np.sum(membership_probs.dot(np.log(cluster_mean_probs)))
    cpt = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                for q in range(n_clusters):
                    for r in range(n_clusters):
                        logb = (adjacency[i, j] * np.log(cluster_transition_probs[q, r])
                                + (1 - adjacency[i, j]) * np.log(1 - cluster_transition_probs[q, r]))
                        cpt += membership_probs[i, q] * membership_probs[j, r] * logb

    return output + cpt / 2 - np.sum(membership_probs * np.log(membership_probs))


def variational_step(adjacency, membership_probs, cluster_mean_probs, cluster_transition_probs, n_clusters: int):
    """Apply the variational step:
    - update membership_probas

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (np.ndarray because of numba, could probably be modified in csr_matrix).
    membership_probs:
        Membership matrix given as a probability over clusters.
    cluster_mean_probs:
        Average value of cluster probability over nodes.
    cluster_transition_probs:
        Probabilities of transition from one cluster to another in one hop.
    n_clusters:
        Number of clusters

    Returns
    -------
    membership_probas:
        membership_probs array updated
    """
    n = adjacency.shape[0]
    log_membership_prob = np.log(np.maximum(membership_probs, eps))
    for i in range(n):
        log_membership_prob[i, :] = np.log(cluster_mean_probs)
        for q in range(n_clusters):
            for j in range(n):
                if j != i:
                    for r in range(n_clusters):
                        log_membership_prob[i, q] += membership_probs[j, r] * \
                                         (adjacency[i, j] * np.log(cluster_transition_probs[q, r])
                                          + (1-adjacency[i, j]) * np.log(1 - cluster_transition_probs[q, r]))

    log_membership_prob = np.maximum(np.minimum(log_membership_prob, xmax), xmin)
    membership_prob = np.exp(log_membership_prob)

    membership_prob = normalize(membership_prob, p=1)

    return np.maximum(membership_prob, eps)


def maximization_step(adjacency, membership_probs, cluster_transition_probs, n_clusters: int):
    """Apply the maximization step:
    - update in place pis
    - update cluster_mean_probas

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (np.ndarray because of numba, could probably be modified in csr_matrix).
    membership_probs:
        Membership matrix given as a probability over clusters.
    cluster_transition_probs:
        Probabilities of transition from one cluster to another in one hop.
    n_clusters:
        Number of clusters

    Returns
    -------
    cluster_mean_probas:
        Alphas array updated
    """
    n = adjacency.shape[0]
    cluster_mean_probs = np.maximum(np.sum(membership_probs, axis=0) / n, eps)

    for cluster_1 in range(n_clusters):
        for cluster_2 in range(n_clusters):
            num = 0
            denom = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        num += membership_probs[i, cluster_1] * membership_probs[j, cluster_2] * adjacency[i, j]
                        denom += membership_probs[i, cluster_1] * membership_probs[j, cluster_2]
            if denom > eps:
                cluster_transition_prob = num / denom
            else:
                # class with a single vertex
                cluster_transition_prob = 0.5

            cluster_transition_probs[cluster_1, cluster_2] = np.minimum(np.maximum(cluster_transition_prob, eps), 1 - eps)

    return cluster_mean_probs


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

        self.cluster_mean_probs = None
        self.membership_probs = None
        self.cluster_transition_probs = None

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

        self.cluster_mean_probs = np.ones(self.n_clusters) / self.n_clusters
        self.cluster_transition_probs = np.zeros((self.n_clusters, self.n_clusters))
        n = adjacency.shape[0]

        if self.init == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, embedding_method=GSVD(self.n_clusters))
            labels = kmeans.fit_transform(adjacency)

            self.membership_probs = np.zeros(shape=(n, self.n_clusters))
            self.membership_probs[:] = np.eye(self.n_clusters)[labels]
        else:
            self.membership_probs = normalize(np.random.rand(n, self.n_clusters), p=1)

        likelihoods = []

        for k in range(self.max_iter):
            self.cluster_mean_probs = maximization_step(adjacency, self.membership_probs, self.cluster_transition_probs,
                                                        self.n_clusters)
            self.membership_probs = variational_step(adjacency, self.membership_probs, self.cluster_mean_probs,
                                                     self.cluster_transition_probs, self.n_clusters)

            likelihoods.append(likelihood(adjacency, self.membership_probs, self.cluster_mean_probs,
                                          self.cluster_transition_probs, self.n_clusters))

            if len(likelihoods) > 1 and (likelihoods[-1] - likelihoods[-2]) < self.tol:
                break

        self.labels_ = np.argmax(self.membership_probs, axis=1)
        self._secondary_outputs(adjacency)

        return self

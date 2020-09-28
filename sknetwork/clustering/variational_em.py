#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2020
@author: Clément Bonet <cbonet@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.clustering.base import BaseClustering
from sknetwork.clustering import KMeans
from sknetwork.embedding import GSVD
from sknetwork.linalg import normalize
from sknetwork.clustering.variational_em_core import likelihood, variational_step


eps = np.finfo(float).eps


def maximization_step(adjacency, membership_probs):
    """Apply the maximization step:
    - update in place cluster_transition_probs
    - update cluster_mean_probas

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (np.ndarray because of numba, could probably be modified in csr_matrix).
    membership_probs:
        Membership matrix given as a probability over clusters.

    Returns
    -------
    cluster_transition_probs:
       Updated probabilities of transition from one cluster to another in one hop.
    """
    n_clusters = membership_probs.shape[1]
    cluster_transition_probs_new = np.zeros((n_clusters, n_clusters))

    for cluster_1 in range(n_clusters):
        for cluster_2 in range(n_clusters):
            num = membership_probs[:, cluster_1].dot(adjacency.dot(membership_probs[:, cluster_2]))
            denom = membership_probs[:, cluster_1].sum() * membership_probs[:, cluster_2].sum() \
                - np.dot(membership_probs[:, cluster_1], membership_probs[:, cluster_2])
            if denom > eps:
                cluster_transition_probs_new[cluster_1, cluster_2] = num / denom
            else:
                # class with a single vertex
                cluster_transition_probs_new[cluster_1, cluster_2] = 0.5

    cluster_transition_probs_new = np.clip(cluster_transition_probs_new, eps, 1 - eps)

    return cluster_transition_probs_new


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
        indptr, indices = adjacency.indptr, adjacency.indices

        n = adjacency.shape[0]

        if self.init == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, embedding_method=GSVD(self.n_clusters))
            labels = kmeans.fit_transform(adjacency)

            membership_probs = np.zeros(shape=(n, self.n_clusters))
            membership_probs[:] = np.eye(self.n_clusters)[labels]
        else:
            membership_probs = normalize(np.random.rand(n, self.n_clusters), p=1)

        likelihood_old, likelihood_new = 0., 0.

        for k in range(self.max_iter):
            cluster_mean_probs = np.maximum(np.mean(membership_probs, axis=0), eps)
            cluster_transition_probs = maximization_step(adjacency, membership_probs)
            membership_probs = variational_step(indptr, indices, membership_probs, cluster_mean_probs,
                                                cluster_transition_probs)

            likelihood_old, likelihood_new = likelihood_new, likelihood(indptr, indices, membership_probs,
                                                                        cluster_mean_probs, cluster_transition_probs)

            if k > 1 and abs(likelihood_new - likelihood_old) < self.tol:
                break

        self.labels_ = np.argmax(membership_probs, axis=1)
        self._secondary_outputs(adjacency)

        return self

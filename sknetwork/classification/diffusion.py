#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Optional

import numpy as np

from sknetwork.classification.base_rank import RankClassifier
from sknetwork.regression.diffusion import Diffusion, Dirichlet
from sknetwork.utils.check import check_labels


def hot_and_cold_seeds(labels_seeds):
    """Make one-vs-all seed labels from seeds.

    Parameters
    ----------
    labels_seeds :

    Returns
    -------
    seeds_all: list
        Personalization vectors.
    """
    seeds_all = []
    classes, _ = check_labels(labels_seeds)

    for label in classes:
        seeds = -np.ones_like(labels_seeds)
        seeds[labels_seeds == label] = 1
        ix = np.logical_and(labels_seeds != label, labels_seeds >= 0)
        seeds[ix] = 0
        seeds_all.append(seeds)

    return seeds_all


def process_scores(scores: np.ndarray, centering: bool) -> np.ndarray:
    """Post-processing of the score matrix.

    Parameters
    ----------
    scores : np.ndarray
        Matrix of scores, shape number of nodes x number of labels.
    centering : bool
        Whether to center temperatures with respect to the mean.
    Returns
    -------
    scores: np.ndarray
    """
    if centering:
        scores -= np.mean(scores, axis=0)
    scores = np.exp(5 * scores)
    return scores


class DiffusionClassifier(RankClassifier):
    """Node classification using multiple diffusions.

    Parameters
    ----------
    n_iter : int
        Number of steps of the diffusion in discrete time (must be positive).
    damping_factor : float (optional)
        Damping factor (default value = 1).
    centering : bool
        Whether to center the temperatures with respect to the mean after diffusion (default = True).
    n_jobs : int
        If positive, number of parallel jobs allowed (-1 means maximum number).
        If ``None``, no parallel computations are made.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_labels,)
        Label of each node.
    membership_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix.
    labels_row_ : np.ndarray
        Labels of rows, for bipartite graphs.
    labels_col_ : np.ndarray
        Labels of columns, for bipartite graphs.
    membership_row_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix of rows, for bipartite graphs.
    membership_col_ : sparse.csr_matrix, shape (n_col, n_labels)
        Membership matrix of columns, for bipartite graphs.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> diffusion = DiffusionClassifier()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = diffusion.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.94

    References
    ----------
    * de Lara, N., & Bonald, T. (2020).
      `A Consistent Diffusion-Based Algorithm for Semi-Supervised Classification on Graphs.
      <https://arxiv.org/pdf/2008.11944.pdf>`
      arXiv preprint arXiv:2008.11944.

    * Zhu, X., Lafferty, J., & Rosenfeld, R. (2005). `Semi-supervised learning with graphs
      <http://pages.cs.wisc.edu/~jerryzhu/machineteaching/pub/thesis.pdf>`
      (Doctoral dissertation, Carnegie Mellon University, language technologies institute, school of computer science).
    """
    def __init__(self, n_iter: int = 10, damping_factor: Optional[float] = None, centering: bool = True,
                 n_jobs: Optional[int] = None):
        algorithm = Diffusion(n_iter, damping_factor)
        super(DiffusionClassifier, self).__init__(algorithm, n_jobs)
        self._process_scores = lambda x: process_scores(x, centering)


class DirichletClassifier(RankClassifier):
    """Node classification using multiple Dirichlet problems.

    Parameters
    ----------
    n_iter : int
        If positive, the solution to the Dirichlet problem is approximated by power iteration for n_iter steps.
        Otherwise, the solution is computed through BiConjugate Stabilized Gradient descent.
    damping_factor : float (optional)
        Damping factor (default value = 1).
    centering : bool
        Whether to center the temperatures with respect to the mean after diffusion (default = True).
    n_jobs : int
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_labels,)
        Label of each node.
    membership_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix.
    labels_row_ : np.ndarray
        Labels of rows, for bipartite graphs.
    labels_col_ : np.ndarray
        Labels of columns, for bipartite graphs.
    membership_row_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix of rows, for bipartite graphs.
    membership_col_ : sparse.csr_matrix, shape (n_col, n_labels)
        Membership matrix of columns, for bipartite graphs.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> dirichlet = DirichletClassifier()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = dirichlet.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    Zhu, X., Lafferty, J., & Rosenfeld, R. (2005). `Semi-supervised learning with graphs
    <http://pages.cs.wisc.edu/~jerryzhu/machineteaching/pub/thesis.pdf>`_
    (Doctoral dissertation, Carnegie Mellon University, language technologies institute, school of computer science).
    """
    def __init__(self, n_iter: int = 10, damping_factor: Optional[float] = None, centering: bool = True,
                 n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = Dirichlet(n_iter, damping_factor, verbose)
        super(DirichletClassifier, self).__init__(algorithm, n_jobs, verbose)
        self._process_seeds = hot_and_cold_seeds
        self._process_scores = lambda x: process_scores(x, centering)

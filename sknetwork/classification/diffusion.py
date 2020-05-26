#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Optional

import numpy as np
from scipy import sparse

from sknetwork.classification.base_rank import RankClassifier, RankBiClassifier
from sknetwork.ranking import Diffusion, Dirichlet
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


def process_scores(scores: np.ndarray) -> np.ndarray:
    """Post-processing of the score matrix.

    Parameters
    ----------
    scores : np.ndarray
        Matrix of scores, shape number of nodes x number of labels.

    Returns
    -------
    scores: np.ndarray
    """
    scores -= np.mean(scores, axis=0)
    scores = np.exp(scores)
    return scores


class DiffusionClassifier(RankClassifier):
    """Node classification using multiple diffusions.

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_iter : int
        Number of steps of the diffusion in discrete time (must be positive).
    damping_factor : float (optional)
        Damping factor (default value = 1).
    n_jobs :
        If positive, number of parallel jobs allowed (-1 means maximum number).
        If ``None``, no parallel computations are made.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node (hard classification).
    membership_ : sparse.csr_matrix
        Membership matrix (soft classification, labels on columns).

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
    Zhu, X., Lafferty, J., & Rosenfeld, R. (2005). `Semi-supervised learning with graphs
    <http://pages.cs.wisc.edu/~jerryzhu/machineteaching/pub/thesis.pdf>`_
    (Doctoral dissertation, Carnegie Mellon University, language technologies institute, school of computer science).
    """
    def __init__(self, n_iter: int = 10, damping_factor: Optional[float] = None, n_jobs: Optional[int] = None):
        algorithm = Diffusion(n_iter, damping_factor)
        super(DiffusionClassifier, self).__init__(algorithm, n_jobs)
        self._process_scores = process_scores


class BiDiffusionClassifier(DiffusionClassifier, RankBiClassifier):
    """Node classification using multiple diffusions.

    * Bigraphs

    Parameters
    ----------
    n_iter : int
        Number of steps of the diffusion in discrete time (must be positive).
    damping_factor : float (optional)
        Damping factor (default value = 1).
    n_jobs :
        If positive, number of parallel jobs allowed (-1 means maximum number).
        If ``None``, no parallel computations are made.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each row.
    labels_row_ : np.ndarray
        Label of each row (copy of **labels_**).
    labels_col_ : np.ndarray
        Label of each column.
    membership_ : sparse.csr_matrix
        Membership matrix of rows (soft classification, labels on columns).
    membership_row_ : sparse.csr_matrix
        Membership matrix of rows (copy of **membership_**).
    membership_col_ : sparse.csr_matrix
        Membership matrix of columns.

    Example
    -------
    >>> from sknetwork.classification import BiDiffusionClassifier
    >>> from sknetwork.data import star_wars
    >>> bidiffusion = BiDiffusionClassifier(n_iter=2)
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1, 2: 0}
    >>> bidiffusion.fit_transform(biadjacency, seeds)
    array([1, 1, 0, 0])
    """
    def __init__(self, n_iter: int = 10, damping_factor: Optional[float] = None, n_jobs: Optional[int] = None):
        super(BiDiffusionClassifier, self).__init__(n_iter=n_iter, damping_factor=damping_factor, n_jobs=n_jobs)


class DirichletClassifier(RankClassifier):
    """Node classification using multiple Dirichlet problems.

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_iter : int
        If positive, the solution to the Dirichlet problem is approximated by power iteration for n_iter steps.
        Otherwise, the solution is computed through BiConjugate Stabilized Gradient descent.
    damping_factor : float (optional)
        Damping factor (default value = 1).
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node (hard classification).
    membership_ : sparse.csr_matrix
        Membership matrix (soft classification, labels on columns).

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
    def __init__(self, n_iter: int = 10, damping_factor: Optional[float] = None, n_jobs: Optional[int] = None,
                 verbose: bool = False):
        algorithm = Dirichlet(n_iter, damping_factor, verbose)
        super(DirichletClassifier, self).__init__(algorithm, n_jobs, verbose)
        self._process_seeds = hot_and_cold_seeds
        self._process_scores = process_scores


class BiDirichletClassifier(DirichletClassifier, RankBiClassifier):
    """Node classification using multiple diffusions.

    * Bigraphs

    Parameters
    ----------
    n_iter : int
        If positive, the solution to the Dirichlet problem is approximated by power iteration for n_iter steps.
        Otherwise, the solution is computed through BiConjugate Stabilized Gradient descent.
    damping_factor : float (optional)
        Damping factor (default value = 1).
    n_jobs :
        If positive, number of parallel jobs allowed (-1 means maximum number).
        If ``None``, no parallel computations are made.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each row.
    labels_row_ : np.ndarray
        Label of each row (copy of **labels_**).
    labels_col_ : np.ndarray
        Label of each column.
    membership_ : sparse.csr_matrix
        Membership matrix of rows (soft classification, labels on columns).
    membership_row_ : sparse.csr_matrix
        Membership matrix of rows (copy of **membership_**).
    membership_col_ : sparse.csr_matrix
        Membership matrix of columns.

    Example
    -------
    >>> from sknetwork.data import star_wars
    >>> bidirichlet = BiDirichletClassifier()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1, 2: 0}
    >>> bidirichlet.fit_transform(biadjacency, seeds)
    array([1, 1, 0, 0])
    """
    def __init__(self, n_iter: int = 10, damping_factor: Optional[float] = None, n_jobs: Optional[int] = None,
                 verbose: bool = False):
        super(BiDirichletClassifier, self).__init__(n_iter=n_iter, damping_factor=damping_factor, verbose=verbose,
                                                    n_jobs=n_jobs)

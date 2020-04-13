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
from sknetwork.ranking import Diffusion
from sknetwork.utils.check import check_labels


def process_seeds(labels_seeds, temperature_max: float = 1):
    """Make one-vs-all seed labels from seeds.

    Parameters
    ----------
    labels_seeds :

    temperature_max
    Returns
    -------
    personalizations: list
        Personalization vectors.

    """

    personalizations = []
    classes, _ = check_labels(labels_seeds)

    for label in classes:
        personalization = -np.ones(labels_seeds.shape[0])
        personalization[labels_seeds == label] = temperature_max
        ix = np.logical_and(labels_seeds != label, labels_seeds >= 0)
        personalization[ix] = 0
        personalizations.append(personalization)

    return personalizations


def process_scores(scores: np.ndarray) -> np.ndarray:
    """Post-processing of the score matrix.

    Parameters
    ----------
    scores
        (n x k) matrix of scores.

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
        If ``n_iter > 0``, apply the diffusion for n_iter steps.
        If ``n_iter <= 0``, use BIConjugate Gradient STABilized iteration to solve the Dirichlet problem.
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
    >>> from sknetwork.classification import DiffusionClassifier
    >>> from sknetwork.data import karate_club
    >>> diffusion = DiffusionClassifier()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = diffusion.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.
    Zhu, X., Lafferty, J., & Rosenfeld, R. (2005). `Semi-supervised learning with graphs
    <http://pages.cs.wisc.edu/~jerryzhu/machineteaching/pub/thesis.pdf>`_
    (Doctoral dissertation, Carnegie Mellon University, language technologies institute, school of computer science).


    """
    def __init__(self, n_iter: int = 10, n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = Diffusion(n_iter, verbose)
        RankClassifier.__init__(self, algorithm, n_jobs, verbose)
        self._process_seeds = process_seeds
        self._process_scores = process_scores


class BiDiffusionClassifier(RankBiClassifier):
    """Node classification using multiple diffusions.

    * Bigraphs

    Parameters
    ----------
    n_iter: int
        If ``n_iter > 0``, apply the diffusion for n_iter steps.
        If ``n_iter <= 0``, use BIConjugate Gradient STABilized iteration to solve the Dirichlet problem.

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
    >>> bidiffusion = BiDiffusionClassifier()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1, 2: 0}
    >>> bidiffusion.fit_transform(biadjacency, seeds)
    array([1, 1, 0, 0])
    """
    def __init__(self, n_iter: int = 10, n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = Diffusion(n_iter, verbose)
        RankBiClassifier.__init__(self, algorithm, n_jobs, verbose)
        self._process_seeds = process_seeds
        self._process_scores = process_scores

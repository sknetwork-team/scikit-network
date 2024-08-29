#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from typing import Optional

import numpy as np

from sknetwork.classification.base_rank import RankClassifier
from sknetwork.ranking.pagerank import PageRank


class PageRankClassifier(RankClassifier):
    """Node classification by multiple personalized PageRanks.

    Parameters
    ----------
    damping_factor: float
        Probability to continue the random walk.
    solver : str
        Which solver to use: 'piteration', 'diteration', 'bicgstab', 'lanczos'.
    n_iter : int
        Number of iterations for some solvers such as ``'piteration'`` or ``'diteration'``.
    tol : float
        Tolerance for the convergence of some solvers such as ``'bicgstab'`` or ``'lanczos'``.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_labels,)
        Labels of nodes.
    probs_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distribution over labels.
    labels_row_ : np.ndarray
        Labels of rows, for bipartite graphs.
    labels_col_ : np.ndarray
        Labels of columns, for bipartite graphs.
    probs_row_ : sparse.csr_matrix, shape (n_row, n_labels)
        Probability distributions over labels of rows, for bipartite graphs.
    probs_col_ : sparse.csr_matrix, shape (n_col, n_labels)
        Probability distributions over labels of columns, for bipartite graphs.

    Example
    -------
    >>> from sknetwork.classification import PageRankClassifier
    >>> from sknetwork.data import karate_club
    >>> pagerank = PageRankClassifier()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> labels = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = pagerank.fit_predict(adjacency, labels)
    >>> float(np.round(np.mean(labels_pred == labels_true), 2))
    0.97

    References
    ----------
    Lin, F., & Cohen, W. W. (2010). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In IEEE International Conference on Advances in Social Networks Analysis and Mining.
    """
    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 0.,
                 n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = PageRank(damping_factor, solver, n_iter, tol)
        super(PageRankClassifier, self).__init__(algorithm, n_jobs, verbose)

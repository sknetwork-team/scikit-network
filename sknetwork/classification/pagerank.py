#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.classification.base_rank import RankClassifier, RankBiClassifier
from sknetwork.linalg.normalization import normalize
from sknetwork.ranking import PageRank
from sknetwork.utils.check import check_seeds


class PageRankClassifier(RankClassifier):
    """Node classification by multiple personalized PageRanks.

    * Graphs
    * Digraphs

    Parameters
    ----------
    damping_factor:
        Probability to continue the random walk.
    solver : :obj:`str`
        Which solver to use: 'piteration', 'diteration', 'bicgstab', 'lanczos'.
    n_iter : int
        Number of iterations for some of the solvers such as ``'piteration'`` or ``'diteration'``.
    tol : float
        Tolerance for the convergence of some solvers such as ``'bicgstab'`` or ``'lanczos'``.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node (hard classification).
    membership_ : sparse.csr_matrix
        Membership matrix (soft classification, columns = labels).

    Example
    -------
    >>> from sknetwork.classification import PageRankClassifier
    >>> from sknetwork.data import karate_club
    >>> pagerank = PageRankClassifier()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = pagerank.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
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


class BiPageRankClassifier(PageRankClassifier, RankBiClassifier):
    """Node classification for bipartite graphs by multiple personalized PageRanks .

    * Bigraphs

    Parameters
    ----------
    damping_factor:
        Probability to continue the random walk.
    solver : :obj:`str`
        Which solver to use: 'piteration', 'diteration', 'bicgstab', 'lanczos'.
    n_iter : int
        Number of iterations for some of the solvers such as ``'piteration'`` or ``'diteration'``.
    tol : float
        Tolerance for the convergence of some solvers such as ``'bicgstab'`` or ``'lanczos'``.

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
    >>> from sknetwork.classification import BiPageRankClassifier
    >>> from sknetwork.data import star_wars
    >>> bipagerank = BiPageRankClassifier()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1, 2: 0}
    >>> bipagerank.fit_transform(biadjacency, seeds)
    array([1, 1, 0, 0])
    """

    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 0.,
                 n_jobs: Optional[int] = None, verbose: bool = False):
        super(BiPageRankClassifier, self).__init__(damping_factor=damping_factor, solver=solver, n_iter=n_iter, tol=tol,
                                                   n_jobs=n_jobs, verbose=verbose)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.classification.base_rank import RankClassifier
from sknetwork.ranking import PageRank
from sknetwork.utils.format import bipartite2undirected
from sknetwork.utils.seeds import stack_seeds


class PageRankClassifier(RankClassifier):
    """Node classification by multiple personalized PageRanks.

    * Graphs
    * Digraphs

    Parameters
    ----------
    damping_factor:
        Damping factor for personalized PageRank.
    solver : str
        Which solver to use: 'spsolve', 'lanczos' (default), 'lsqr' or 'halko'.
        Otherwise, the random walk is emulated for a certain number of iterations.
    n_iter : int
        If ``solver`` is not one of the standard values, the pagerank is approximated by emulating the random walk for
        ``n_iter`` iterations.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node (hard classification).
    membership_ : sparse.csr_matrix
        Membership matrix (soft classification, columns = labels).

    Example
    -------
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
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """

    def __init__(self, damping_factor: float = 0.85, solver: str = 'bicgstab', n_iter: int = 10,
                 n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = PageRank(damping_factor, solver, n_iter)
        super(PageRankClassifier, self).__init__(algorithm, n_jobs, verbose)

        self.labels_ = None
        self.membership_ = None


class BiPageRankClassifier(PageRankClassifier):
    """Node classification for bipartite graphs by multiple personalized PageRanks .

    * Bigraphs

    Parameters
    ----------
    damping_factor:
        Damping factor for personalized PageRank.
    solver : str
        Which solver to use: 'spsolve', 'lanczos' (default), 'lsqr' or 'halko'.
        Otherwise, the random walk is emulated for a certain number of iterations.
    n_iter : int
        If ``solver`` is not one of the standard values, the pagerank is approximated by emulating the random walk for
        ``n_iter`` iterations.

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
    >>> bipagerank = BiPageRankClassifier()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1, 2: 0}
    >>> bipagerank.fit_transform(biadjacency, seeds)
    array([1, 1, 0, 0])
    """

    def __init__(self, damping_factor: float = 0.85, solver: str = None, n_iter: int = 10,
                 n_jobs: Optional[int] = None, verbose: bool = False):
        PageRankClassifier.__init__(self, damping_factor, solver, n_iter, n_jobs, verbose)

        self.labels_ = None
        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_ = None
        self.membership_row_ = None
        self.membership_col_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            seeds_row: Union[np.ndarray, dict], seeds_col: Union[np.ndarray, dict, None] = None) -> 'RankClassifier':
        """Compute labels.

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix of the graph.
        seeds_row :
            Seed rows. Can be a dict {node: label} or an array where "-1" means no label.
        seeds_col :
            Seed columns (optional). Same format.

        Returns
        -------
        self: :class:`BiPageRankClassifier`
        """
        n_row, n_col = biadjacency.shape
        labels = stack_seeds(n_row, n_col, seeds_row, seeds_col)
        adjacency = bipartite2undirected(biadjacency)
        PageRankClassifier.fit(self, adjacency, labels)

        self.labels_row_ = self.labels_[:n_row]
        self.labels_col_ = self.labels_[n_row:]
        self.labels_ = self.labels_row_
        self.membership_row_ = self.membership_[:n_row]
        self.membership_col_ = self.membership_[n_row:]
        self.membership_ = self.membership_row_

        return self

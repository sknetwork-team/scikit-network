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
from sknetwork.ranking import BiPageRank, PageRank


class PageRankClassifier(RankClassifier):
    """Node classification by multiple personalized PageRanks.

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


class BiPageRankClassifier(RankClassifier):
    """Node classification for bipartite graphs by multiple personalized PageRanks .

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
    >>> from sknetwork.data import star_wars
    >>> bipagerank = BiPageRankClassifier()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1, 2: 0}
    >>> bipagerank.fit_transform(biadjacency, seeds)
    array([1, 1, 0, 0])
    """

    def __init__(self, damping_factor: float = 0.85, solver: str = None, n_iter: int = 10,
                 n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = BiPageRank(damping_factor, solver, n_iter)
        super(BiPageRankClassifier, self).__init__(algorithm, n_jobs, verbose)

        self.labels_row_ = None
        self.membership_row_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            seeds_row: Union[np.ndarray, dict]) -> 'RankClassifier':
        """Compute labels."""
        RankClassifier.fit(self, biadjacency, seeds_row)
        self.labels_row_ = self.labels_
        self.membership_row_ = self.membership_

        return self

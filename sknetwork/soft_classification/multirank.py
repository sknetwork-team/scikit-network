#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from functools import partial
from multiprocessing import Pool
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.ranking import PageRank, BiPageRank
from sknetwork.soft_classification.base import BaseSoftClassifier
from sknetwork.utils.checks import check_seeds, check_labels, check_n_jobs


class MultiRank(BaseSoftClassifier):
    """Semi-Supervised classification based on personalized PageRank.

    Parameters
    ----------
    damping_factor:
        Damping factor for personalized PageRank.
    solver:
        Which solver to use for PageRank.
    n_jobs:
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.

    Attributes
    ----------
    membership_: np.ndarray
        Component (i, k) indicates the level of membership of node i in the k-th cluster.
        If the provided labels are not consecutive integers starting from 0,
        the k-th column of the membership corresponds to the k-th label in ascending order.
        The rows are normalized to sum to 1.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> multirank = MultiRank()
    >>> adjacency, labels_true = karate_club(return_labels=True)
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> membership_ = multirank.fit_transform(adjacency, seeds)
    >>> membership_.shape
    (34, 2)

    References
    ----------
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """

    def __init__(self, damping_factor: float = 0.85, solver: str = 'lanczos', n_jobs: Optional[int] = None):
        super(MultiRank, self).__init__()

        self.damping_factor = damping_factor
        self.solver = solver
        self.n_jobs = check_n_jobs(n_jobs)

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'MultiRank':
        """Compute personalized PageRank using each given labels as seed set.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.
        seeds: Dict or ndarray,
            If dict, ``(key, val)`` indicates that node ``key`` has label ``val``.
            If ndarray, ``seeds[i] = val`` indicates that node ``i`` has label ``val``.
            Negative values are treated has no label.

        Returns
        -------
        self: :class:`MultiRank`

        """
        if isinstance(self, BiMultiRank):
            pr = BiPageRank(self.damping_factor, self.solver)
        else:
            pr = PageRank(self.damping_factor, self.solver)

        seeds_labels = check_seeds(seeds, adjacency)
        classes, n_classes = check_labels(seeds_labels)

        n: int = adjacency.shape[0]
        personalizations = []
        for label in classes:
            personalization = np.array(seeds_labels == label).astype(int)
            personalizations.append(personalization)

        if self.n_jobs != 1:
            local_function = partial(pr.fit_transform, adjacency)
            with Pool(self.n_jobs) as pool:
                membership = np.array(pool.map(local_function, personalizations))
            membership = membership.T
        else:
            membership = np.zeros((n, n_classes))
            for i in range(n_classes):
                membership[:, i] = pr.fit_transform(adjacency, personalization=personalizations[i])[:n]

        norm = np.sum(membership, axis=1)
        membership[norm > 0] /= norm[norm > 0, np.newaxis]

        self.membership_ = membership
        return self


class BiMultiRank(MultiRank):
    """Semi-Supervised clustering based on personalized PageRank for bipartite graphs.
    See :class:`sknetwork.ranking.BiPageRank`
    """

    def __init__(self, damping_factor: float = 0.85, solver: str = 'lanczos', n_jobs: Optional[int] = None):
        super(BiMultiRank, self).__init__(damping_factor, solver, n_jobs)

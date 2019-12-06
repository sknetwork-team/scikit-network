#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.classification import BaseClassifier
from sknetwork.soft_classification import MultiRank, BiMultiRank


class MaxRank(BaseClassifier):
    """Semi-supervised node classification using multiple personalized PageRanks.
    See :class:`sknetwork.soft_classification.MultiRank`.

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
    labels_: ndarray
        Predicted label of each node.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> maxrank = MaxRank()
    >>> adjacency, labels_true = karate_club(return_labels=True)
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = maxrank.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """

    def __init__(self, damping_factor: float = 0.85, solver: str = 'lanczos', n_jobs: Optional[int] = None):
        super(MaxRank, self).__init__()
        self.damping_factor = damping_factor
        self.solver = solver
        self.n_jobs = n_jobs

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'MaxRank':
        """Compute personalized PageRank using each given label as a seed set.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.
        seeds: Dict or ndarray,
            If dict, ``(key, val)`` indicates that node ``key`` has label ``val``.
            If ndarray, ``seeds[i] = val`` indicates that node ``i`` has label ``val``.
            Negative values are treated as no label.

        Returns
        -------
        self: :class:`MaxRank`

        """
        if isinstance(self, BiMaxRank):
            multirank = BiMultiRank(self.damping_factor, self.solver, self.n_jobs)
        else:
            multirank = MultiRank(self.damping_factor, self.solver, self.n_jobs)

        if isinstance(seeds, np.ndarray):
            unique_labels = np.unique(seeds[seeds >= 0])
        elif isinstance(seeds, dict):
            unique_labels = np.array([val for val in sorted(set(seeds.values())) if val >= 0])
        else:
            raise TypeError('"seeds" must be a dictionary or a one-dimensional array.')

        multirank.fit(adjacency, seeds)
        labels = np.argmax(multirank.membership_, axis=1)

        self.labels_ = np.array([unique_labels[val] for val in labels])

        return self


class BiMaxRank(MaxRank):
    """Semi-supervised node classification using multiple personalized PageRanks for bipartite graphs.
    See :class:`sknetwork.soft_classification.BiMultiRank`.

    """

    def __init__(self, damping_factor: float = 0.85, solver: str = 'lanczos', n_jobs: Optional[int] = None):
        MaxRank.__init__(self, damping_factor, solver, n_jobs)

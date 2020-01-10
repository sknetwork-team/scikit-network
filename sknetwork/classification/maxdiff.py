#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.classification import BaseClassifier
from sknetwork.soft_classification import MultiDiff, BiMultiDiff
from sknetwork.utils.checks import check_seeds, check_labels


class MaxDiff(BaseClassifier):
    """Semi-supervised node classification using multiple diffusions.
    See :class:`sknetwork.soft_classification.MultiDiff`.

    Parameters
    ----------

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
    >>> maxdiff = MaxDiff()
    >>> adjacency, labels_true = karate_club(return_labels=True)
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = maxdiff.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """

    def __init__(self, verbose: bool = False, n_iter: int = 0, n_jobs: Optional[int] = None):
        super(MaxDiff, self).__init__()
        self.verbose = verbose
        self.n_iter = n_iter
        self.n_jobs = n_jobs

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'MaxDiff':
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
        self: :class:`MaxDiff`

        """
        if isinstance(self, BiMaxDiff):
            multidiff = BiMultiDiff(self.verbose, self.n_iter, self.n_jobs)
        else:
            multidiff = MultiDiff(self.verbose, self.n_iter, self.n_jobs)

        seeds_labels = check_seeds(seeds, adjacency)
        classes, n_classes = check_labels(seeds_labels)

        multidiff.fit(adjacency, seeds)
        labels = np.argmax(multidiff.membership_, axis=1)

        self.labels_ = np.array([classes[val] for val in labels]).astype(int)

        return self


class BiMaxDiff(MaxDiff):
    """Semi-supervised node classification using multiple personalized PageRanks for bipartite graphs.
    See :class:`sknetwork.soft_classification.BiMultiRank`.

    """

    def __init__(self, verbose: bool = False, n_iter: int = 0, n_jobs: Optional[int] = None):
        super(BiMaxDiff, self).__init__(verbose, n_iter, n_jobs)

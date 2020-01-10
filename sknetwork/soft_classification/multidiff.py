#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""

from functools import partial
from multiprocessing import Pool
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.ranking import Diffusion, BiDiffusion
from sknetwork.soft_classification.base import BaseSoftClassifier
from sknetwork.utils.checks import check_seeds, check_labels, check_n_jobs
from sknetwork.utils.verbose import VerboseMixin


class MultiDiff(BaseSoftClassifier, VerboseMixin):
    """Semi-Supervised classification based on graph diffusion.

    Parameters
    ----------
    verbose: bool
        Verbose mode.
    n_iter: int
        If ``n_iter > 0``, the algorithm will emulate the diffusion for n_iter steps.
        If ``n_iter <= 0``, the algorithm will use BIConjugate Gradient STABilized iteration
        to solve the Dirichlet problem.
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
    >>> multidiff = MultiDiff()
    >>> adjacency, labels_true = karate_club(return_labels=True)
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> membership_ = multidiff.fit_transform(adjacency, seeds)
    >>> membership_.shape
    (34, 2)

    """

    def __init__(self, verbose: bool = False, n_iter: int = 0, n_jobs: Optional[int] = None):
        super(MultiDiff, self).__init__()
        VerboseMixin.__init__(self, verbose)

        self.verbose = verbose
        self.n_iter = n_iter
        self.n_jobs = check_n_jobs(n_jobs)

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'MultiDiff':
        """Compute multiple diffusions in a 1-vs-all mode. One class is the hot seeds
        while all the others are cold seeds.

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
        self: :class:`MultiDiff`

        """

        if isinstance(self, BiMultiDiff):
            diffusion = BiDiffusion(self.verbose, self.n_iter)
        else:
            diffusion = Diffusion(self.verbose, self.n_iter)

        seeds_labels = check_seeds(seeds, adjacency)
        classes, n_classes = check_labels(seeds_labels)

        n: int = adjacency.shape[0]
        personalizations = []
        for label in classes:
            personalization = -np.ones(n)
            personalization[seeds_labels == label] = 1
            ix = np.logical_and(seeds_labels != label, seeds_labels >= 0)
            personalization[ix] = 0
            personalizations.append(personalization)

        if self.n_jobs != 1:
            local_function = partial(diffusion.fit_transform, adjacency)
            with Pool(self.n_jobs) as pool:
                membership = np.array(pool.map(local_function, personalizations))
            membership = membership.T
        else:
            membership = np.zeros((n, n_classes))
            for i in range(n_classes):
                membership[:, i] = diffusion.fit_transform(adjacency, personalization=personalizations[i])[:n]

        membership -= np.mean(membership, axis=0)
        membership = np.exp(membership)

        norms = membership.sum(axis=1)
        ix = np.argwhere(norms == 0).ravel()
        if len(ix) > 0:
            self.log.print('Nodes ', ix, ' have a null membership.')
        membership[~ix] /= norms[~ix, np.newaxis]

        self.membership_ = membership
        return self


class BiMultiDiff(MultiDiff):
    """Semi-Supervised classification based on graph diffusion for bipartite graphs.
    """

    def __init__(self, verbose: bool = False, n_iter: int = 0, n_jobs: Optional[int] = None):
        super(BiMultiDiff, self).__init__(verbose, n_iter, n_jobs)

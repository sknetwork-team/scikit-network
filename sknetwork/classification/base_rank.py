#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""
from functools import partial
from multiprocessing import Pool
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.classification import BaseClassifier, BaseBiClassifier
from sknetwork.linalg.normalization import normalize
from sknetwork.ranking import BaseRanking
from sknetwork.utils.seeds import stack_seeds
from sknetwork.utils.check import check_seeds, check_labels, check_n_jobs
from sknetwork.utils.format import bipartite2undirected
from sknetwork.utils.verbose import VerboseMixin


class RankClassifier(BaseClassifier, VerboseMixin):
    """Generic class for ranking based classifiers.

    Parameters
    ----------
    algorithm :
        Which ranking algorithm to use.
    n_jobs :
        If positive, number of parallel jobs allowed (-1 means maximum number).
        If ``None``, no parallel computations are made.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node.
    membership_ : sparse.csr_matrix
        Membership matrix (labels = columns).
    """
    def __init__(self, algorithm: BaseRanking, n_jobs: Optional[int] = None, verbose: bool = False):
        super(RankClassifier, self).__init__()
        VerboseMixin.__init__(self, verbose)

        self.algorithm = algorithm
        self.n_jobs = check_n_jobs(n_jobs)
        self.verbose = verbose

    @staticmethod
    def _process_seeds(labels_seeds: np.ndarray) -> list:
        """Make one-vs-all seed labels from seeds.

        Parameters
        ----------
        labels_seeds

        Returns
        -------
        List of seeds vectors.
        """
        seeds_all = []
        classes, _ = check_labels(labels_seeds)

        for label in classes:
            seeds = np.array(labels_seeds == label).astype(int)
            seeds_all.append(seeds)

        return seeds_all

    @staticmethod
    def _process_scores(scores: np.ndarray) -> np.ndarray:
        """Post-processing of the membership matrix.

        Parameters
        ----------
        scores
            Matrix of scores, shape number of nodes x number of labels.

        Returns
        -------
        scores: np.ndarray
        """
        return scores

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'RankClassifier':
        """Fit algorithm to the data.

        Parameters
        ----------
        adjacency:
            Adjacency matrix of the graph.
        seeds:
            Seed nodes (labels as dictionary or array; negative values ignored).

        Returns
        -------
        self: :class:`RankClassifier`
        """
        n = adjacency.shape[0]
        seeds_labels = check_seeds(seeds, n).astype(int)
        classes, n_classes = check_labels(seeds_labels)

        seeds_all = self._process_seeds(seeds_labels)
        local_function = partial(self.algorithm.fit_transform, adjacency)
        with Pool(self.n_jobs) as pool:
            scores = np.array(pool.map(local_function, seeds_all))
        scores = scores.T

        scores = self._process_scores(scores)
        scores = normalize(scores)

        membership = sparse.coo_matrix(scores)
        membership.col = classes[membership.col]

        labels = np.argmax(scores, axis=1)
        self.labels_ = classes[labels]
        self.membership_ = sparse.csr_matrix(membership, shape=(n, np.max(seeds_labels) + 1))

        return self


class RankBiClassifier(RankClassifier, BaseBiClassifier):
    """Generic class for ranking based classifiers on bipartite graphs."""
    def __init__(self, algorithm: BaseRanking, n_jobs: Optional[int] = None, verbose: bool = False):
        super(RankBiClassifier, self).__init__(algorithm=algorithm, n_jobs=n_jobs, verbose=verbose)

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray],
            seeds_row: Union[np.ndarray, dict], seeds_col: Union[np.ndarray, dict, None] = None) -> 'RankClassifier':
        """Compute labels.

        Parameters
        ----------
        biadjacency :
            Biadjacency matrix of the graph.
        seeds_row :
            Seed rows (labels as dictionary or array; negative values ignored).
        seeds_col :
            Seed columns (optional). Same format.

        Returns
        -------
        self: :class:`BiPageRankClassifier`
        """
        n_row, n_col = biadjacency.shape
        labels = stack_seeds(n_row, n_col, seeds_row, seeds_col)
        adjacency = bipartite2undirected(biadjacency)
        RankClassifier.fit(self, adjacency, labels)

        self.labels_row_ = self.labels_[:n_row]
        self.labels_col_ = self.labels_[n_row:]
        self.labels_ = self.labels_row_
        self.membership_row_ = self.membership_[:n_row]
        self.membership_col_ = self.membership_[n_row:]
        self.membership_ = self.membership_row_

        return self

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

from sknetwork.classification.base import BaseClassifier
from sknetwork.linalg.normalization import normalize
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_labels, check_n_jobs
from sknetwork.utils.format import get_adjacency_seeds
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
    labels_ : np.ndarray, shape (n_labels,)
        Label of each node.
    membership_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix.
    labels_row_ : np.ndarray
        Labels of rows, for bipartite graphs.
    labels_col_ : np.ndarray
        Labels of columns, for bipartite graphs.
    membership_row_ : sparse.csr_matrix, shape (n_row, n_labels)
        Membership matrix of rows, for bipartite graphs.
    membership_col_ : sparse.csr_matrix, shape (n_col, n_labels)
        Membership matrix of columns, for bipartite graphs.
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
        scores : np.ndarray
        """
        return scores

    def _split_vars(self, shape):
        """Split the vector of labels and build membership matrix."""
        n_row = shape[0]
        self.labels_row_ = self.labels_[:n_row]
        self.labels_col_ = self.labels_[n_row:]
        self.labels_ = self.labels_row_
        self.membership_row_ = self.membership_[:n_row]
        self.membership_col_ = self.membership_[n_row:]
        self.membership_ = self.membership_row_

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict] = None,
            seeds_row: Union[np.ndarray, dict] = None, seeds_col: Union[np.ndarray, dict] = None) -> 'RankClassifier':
        """Fit algorithm to data.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        seeds :
            Seed nodes (labels as dictionary or array; negative values ignored).
        seeds_row, seeds_col :
            Seed rows and columns (for bipartite graphs).
        Returns
        -------
        self: :class:`RankClassifier`
        """
        adjacency, seeds_labels, bipartite = get_adjacency_seeds(input_matrix, seeds=seeds, seeds_row=seeds_row,
                                                                 seeds_col=seeds_col)
        seeds_labels = seeds_labels.astype(int)
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
        self.membership_ = sparse.csr_matrix(membership, shape=(adjacency.shape[0], np.max(seeds_labels) + 1))

        if bipartite:
            self._split_vars(input_matrix.shape)

        return self

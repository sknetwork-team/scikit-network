#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
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
from sknetwork.utils.format import get_adjacency_values
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
    def _process_labels(labels: np.ndarray) -> list:
        """Make one-vs-all binary labels from labels.

        Parameters
        ----------
        labels

        Returns
        -------
        List of binary labels.
        """
        labels_all = []
        labels_unique, _ = check_labels(labels)

        for label in labels_unique:
            labels_binary = np.array(labels == label).astype(int)
            labels_all.append(labels_binary)

        return labels_all

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

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray], labels: Union[np.ndarray, dict] = None,
            labels_row: Union[np.ndarray, dict] = None, labels_col: Union[np.ndarray, dict] = None) -> 'RankClassifier':
        """Fit algorithm to data.

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        labels :
            Known labels (dictionary or array; negative values ignored).
        labels_row, labels_col :
            Known labels on rows and columns (for bipartite graphs).
        Returns
        -------
        self: :class:`RankClassifier`
        """
        adjacency, seeds_labels, bipartite = get_adjacency_values(input_matrix, values=labels, values_row=labels_row,
                                                                  values_col=labels_col)
        seeds_labels = seeds_labels.astype(int)
        labels_unique, n_classes = check_labels(seeds_labels)
        seeds_all = self._process_labels(seeds_labels)
        local_function = partial(self.algorithm.fit_transform, adjacency)
        with Pool(self.n_jobs) as pool:
            scores = np.array(pool.map(local_function, seeds_all))
        scores = scores.T

        scores = self._process_scores(scores)
        scores = normalize(scores)

        membership = sparse.coo_matrix(scores)
        membership.col = labels_unique[membership.col]

        labels = np.argmax(scores, axis=1)
        self.labels_ = labels_unique[labels]
        self.membership_ = sparse.csr_matrix(membership, shape=(adjacency.shape[0], np.max(seeds_labels) + 1))
        self._split_vars(input_matrix.shape)

        return self

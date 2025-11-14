#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in March 2020
@author: Nathan de Lara <nathan.delara@polytechnique.org>
"""
from functools import partial
from multiprocessing import Pool
from typing import Union, Optional

import numpy as np
from scipy import sparse

from sknetwork.classification.base import BaseClassifier
from sknetwork.linalg.normalizer import normalize
from sknetwork.ranking.base import BaseRanking
from sknetwork.utils.check import check_labels, check_n_jobs
from sknetwork.utils.format import get_adjacency_values


class RankClassifier(BaseClassifier):
    """Generic class for ranking based classifiers.

    Parameters
    ----------
    algorithm :
        Which ranking algorithm to use.
    n_jobs :
        If positive, number of parallel jobs allowed (-1 means maximum number).
        If ``None``, no parallel computations are made.

    Attributes
    ----------
    labels\_ : np.ndarray, shape (n_nodes,)
        Label of each node.
    probs\_ : sparse.csr_matrix, shape (n_nodes, n_labels)
        Probability distribution over labels.
    """
    def __init__(self, algorithm: BaseRanking, n_jobs: Optional[int] = None, verbose: bool = False):
        super(RankClassifier, self).__init__()

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
        """Post-processing of the scores.

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
        self.probs_row_ = self.probs_[:n_row]
        self.probs_col_ = self.probs_[n_row:]
        self.probs_ = self.probs_row_

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
        local_function = partial(self.algorithm.fit_predict, adjacency)
        with Pool(self.n_jobs) as pool:
            scores = np.array(pool.map(local_function, seeds_all))
        scores = scores.T

        scores = self._process_scores(scores)
        scores = normalize(scores)

        probs = sparse.coo_matrix(scores)
        probs.col = labels_unique[probs.col]

        labels = np.argmax(scores, axis=1)
        self.labels_ = labels_unique[labels]
        self.probs_ = sparse.csr_matrix(probs, shape=(adjacency.shape[0], np.max(seeds_labels) + 1))
        self._split_vars(input_matrix.shape)

        return self

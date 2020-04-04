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

from sknetwork.classification import BaseClassifier
from sknetwork.ranking import BaseRanking
from sknetwork.utils.check import check_seeds, check_labels
from sknetwork.utils.verbose import VerboseMixin


class RankClassifier(BaseClassifier, VerboseMixin):
    """Generic class for ranking based classifiers.

    Parameters
    ----------
    algorithm :
        Which ranking algorithm to use.
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
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
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def _process_seeds(labels_seeds: np.ndarray) -> list:
        """Make one-vs-all seed labels from seeds.

        Parameters
        ----------
        labels_seeds

        Returns
        -------
        personalizations
            List of personalization vectors.

        """

        personalizations = []
        classes, _ = check_labels(labels_seeds)

        for label in classes:
            personalization = np.array(labels_seeds == label).astype(int)
            personalizations.append(personalization)

        return personalizations

    @staticmethod
    def _process_scores(scores: np.ndarray) -> np.ndarray:
        """Post-processing of the membership matrix.

        Parameters
        ----------
        scores
            (n x k) matrix of scores.

        Returns
        -------
        scores: np.ndarray

        """
        return scores

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]) -> 'RankClassifier':
        """

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

        personalizations = self._process_seeds(seeds_labels)

        if self.n_jobs != 1:
            local_function = partial(self.algorithm.fit_transform, adjacency)
            with Pool(self.n_jobs) as pool:
                scores = np.array(pool.map(local_function, personalizations))
            scores = scores.T
        else:
            scores = np.zeros((n, n_classes))
            for i in range(n_classes):
                scores[:, i] = self.algorithm.fit_transform(adjacency, personalization=personalizations[i])[:n]

        scores = self._process_scores(scores)

        sums = np.sum(scores, axis=1)
        ix = (sums == 0)
        if ix.sum() > 0:
            self.log.print('Some nodes have no label.')
        scores[~ix] /= sums[~ix, np.newaxis]

        membership = sparse.coo_matrix(scores)
        membership.col = classes[membership.col]

        labels = np.argmax(scores, axis=1)
        self.labels_ = classes[labels]
        self.membership_ = sparse.csr_matrix(membership, shape=(n, np.max(seeds_labels) + 1))

        return self

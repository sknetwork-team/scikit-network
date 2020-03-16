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
from sknetwork.utils.checks import check_seeds, check_labels
from sknetwork.utils.verbose import VerboseMixin


class RankClassifier(BaseClassifier, VerboseMixin):
    """Generic class for ranking based classifiers.

    Parameters
    ----------
    algorithm:
        Which ranking algorithm to use.
    n_jobs:
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.

    """

    def __init__(self, algorithm: BaseRanking, n_jobs: Optional[int] = None, verbose: bool = False):
        super(RankClassifier, self).__init__()
        VerboseMixin.__init__(self, verbose)

        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.membership_ = None

    @staticmethod
    def process_seeds(seeds_labels: np.ndarray) -> list:
        """Make one-vs-all seed labels from seeds.

        Parameters
        ----------
        seeds_labels

        Returns
        -------
        personalizations
            List of personalization vectors.

        """

        personalizations = []
        classes, _ = check_labels(seeds_labels)

        for label in classes:
            personalization = np.array(seeds_labels == label).astype(int)
            personalizations.append(personalization)

        return personalizations

    @staticmethod
    def process_membership(membership: np.ndarray) -> np.ndarray:
        """Post-processing of the membership matrix.

        Parameters
        ----------
        membership
            (n x k) matrix of membership.

        Returns
        -------
        membership: np.ndarray

        """
        return membership

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: Union[np.ndarray, dict]):
        """

        Parameters
        ----------
        adjacency
        seeds

        Returns
        -------

        """
        seeds_labels = check_seeds(seeds, adjacency)
        classes, n_classes = check_labels(seeds_labels)
        n: int = adjacency.shape[0]

        personalizations = self.process_seeds(seeds_labels)

        if self.n_jobs != 1:
            local_function = partial(self.algorithm.fit_transform, adjacency)
            with Pool(self.n_jobs) as pool:
                membership = np.array(pool.map(local_function, personalizations))
            membership = membership.T
        else:
            membership = np.zeros((n, n_classes))
            for i in range(n_classes):
                membership[:, i] = self.algorithm.fit_transform(adjacency, personalization=personalizations[i])[:n]

        membership = self.process_membership(membership)

        norms = membership.sum(axis=1)
        ix = np.argwhere(norms == 0).ravel()
        if len(ix) > 0:
            self.log.print('Nodes ', ix, ' have a null membership.')
        membership[~ix] /= norms[~ix, np.newaxis]

        labels = np.argmax(membership, axis=1)
        self.labels_ = np.array([classes[val] for val in labels]).astype(int)
        self.membership_ = membership

        return self

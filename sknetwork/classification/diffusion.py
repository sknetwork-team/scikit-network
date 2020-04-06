#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.classification.base_rank import RankClassifier
from sknetwork.ranking import Diffusion
from sknetwork.utils.check import check_labels
from sknetwork.utils.format import bipartite2undirected
from sknetwork.utils.seeds import stack_seeds


class DiffusionClassifier(RankClassifier):
    """Node classification using multiple diffusions.

    Parameters
    ----------
    n_iter: int
        If ``n_iter > 0``, apply the diffusion for n_iter steps.
        If ``n_iter <= 0``, use BIConjugate Gradient STABilized iteration to solve the Dirichlet problem.
    n_jobs :
        If an integer value is given, denotes the number of workers to use (-1 means the maximum number will be used).
        If ``None``, no parallel computations are made.
    verbose :
        Verbose mode.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node (hard classification).
    membership_ : sparse.csr_matrix
        Membership matrix (soft classification, columns = labels).

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> diff = DiffusionClassifier()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = diff.fit_transform(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """
    def __init__(self, n_iter: int = 10, n_jobs: Optional[int] = None, verbose: bool = False):
        algorithm = Diffusion(n_iter, verbose)
        super(DiffusionClassifier, self).__init__(algorithm, n_jobs, verbose)

    @staticmethod
    def _process_seeds(labels_seeds, temperature_max: float = 5):
        """Make one-vs-all seed labels from seeds.

        Parameters
        ----------
        labels_seeds :

        temperature_max
        Returns
        -------
        personalizations: list
            Personalization vectors.

        """

        personalizations = []
        classes, _ = check_labels(labels_seeds)

        for label in classes:
            personalization = -np.ones(labels_seeds.shape[0])
            personalization[labels_seeds == label] = temperature_max
            ix = np.logical_and(labels_seeds != label, labels_seeds >= 0)
            personalization[ix] = 0
            personalizations.append(personalization)

        return personalizations

    @staticmethod
    def _process_scores(scores: np.ndarray):
        """Post-processing of the score matrix.

        Parameters
        ----------
        scores
            (n x k) matrix of scores.

        Returns
        -------
        scores: np.ndarray

        """
        scores -= np.mean(scores, axis=0)
        scores = np.exp(scores)
        return scores


class BiDiffusionClassifier(DiffusionClassifier):
    """Node classification using multiple diffusions.

    Parameters
    ----------
    n_iter: int
        If ``n_iter > 0``, apply the diffusion for n_iter steps.
        If ``n_iter <= 0``, use BIConjugate Gradient STABilized iteration to solve the Dirichlet problem.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node (hard classification).
    membership_ : sparse.csr_matrix
        Membership matrix (soft classification, columns = labels).

    Example
    -------
    >>> from sknetwork.data import star_wars
    >>> bidiff = BiDiffusionClassifier()
    >>> biadjacency = star_wars()
    >>> seeds = {0: 1, 2: 0}
    >>> bidiff.fit_transform(biadjacency, seeds)
    array([1, 1, 0, 0])

    References
    ----------
    Lin, F., & Cohen, W. W. (2010, August). `Semi-supervised classification of network data using very few labels.
    <https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf>`_
    In 2010 International Conference on Advances in Social Networks Analysis and Mining (pp. 192-199). IEEE.

    """
    def __init__(self, n_iter: int = 10, n_jobs: Optional[int] = None, verbose: bool = False):
        DiffusionClassifier.__init__(self, n_iter, n_jobs, verbose)

        self.labels_row_ = None
        self.labels_col_ = None
        self.membership_row_ = None
        self.membership_col_ = None

    def fit(self, biadjacency: Union[sparse.csr_matrix, np.ndarray], seeds_row: Union[np.ndarray, dict],
            seeds_col: Optional[Union[np.ndarray, dict]] = None) -> 'RankClassifier':
        """Jointly classify rows and columns."""

        n_row, n_col = biadjacency.shape
        labels = stack_seeds(n_row, n_col, seeds_row, seeds_col)
        adjacency = bipartite2undirected(biadjacency)
        DiffusionClassifier.fit(self, adjacency, labels)

        self.labels_row_ = self.labels_[:n_row]
        self.labels_col_ = self.labels_[n_row:]
        self.membership_row_ = self.membership_[:n_row]
        self.membership_col_ = self.membership_[n_row:]
        self.labels_ = self.labels_row_
        self.membership_ = self.membership_row_

        return self

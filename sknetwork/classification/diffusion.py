#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar, 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Optional

import numpy as np

from sknetwork.classification.base_rank import RankClassifier
from sknetwork.ranking import BiDiffusion, Diffusion
from sknetwork.utils.checks import check_labels


class DiffusionClassifier(RankClassifier):
    """Node classification using multiple diffusions.

    Parameters
    ----------
    n_iter: int
        If ``n_iter > 0``, the algorithm will emulate the diffusion for n_iter steps.
        If ``n_iter <= 0``, the algorithm will use BIConjugate Gradient STABilized iteration
        to solve the Dirichlet problem.
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
    >>> adjacency, labels_true = karate_club(return_labels=True)
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

        self.labels_ = None
        self.membership_ = None

    @staticmethod
    def _process_seeds(seeds_labels, temperature_max: float = 5):
        """Make one-vs-all seed labels from seeds.

        Parameters
        ----------
        seeds_labels :

        temperature_max
        Returns
        -------
        personalizations: list
            Personalization vectors.

        """

        personalizations = []
        classes, _ = check_labels(seeds_labels)

        for label in classes:
            personalization = -np.ones(seeds_labels.shape[0])
            personalization[seeds_labels == label] = temperature_max
            ix = np.logical_and(seeds_labels != label, seeds_labels >= 0)
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
        If ``n_iter > 0``, the algorithm will emulate the diffusion for n_iter steps.
        If ``n_iter <= 0``, the algorithm will use BIConjugate Gradient STABilized iteration
        to solve the Dirichlet problem.

    Attributes
    ----------
    labels_ : np.ndarray
        Label of each node (hard classification).
    membership_ : sparse.csr_matrix
        Membership matrix (soft classification, columns = labels).

    Example
    -------
    >>> from sknetwork.data import star_wars_villains
    >>> bidiff = BiDiffusionClassifier()
    >>> biadjacency = star_wars_villains()
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
        algorithm = BiDiffusion(n_iter, verbose)
        RankClassifier.__init__(self, algorithm, n_jobs, verbose)

        self.labels_ = None
        self.membership_ = None

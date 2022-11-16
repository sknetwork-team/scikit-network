#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2022
@author: Thomas Bonald <thomas.bonald@telecom-paris.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.classification.base import BaseClassifier
from sknetwork.linalg.normalization import normalize
from sknetwork.utils.format import get_adjacency_seeds
from sknetwork.utils.membership import get_membership
from sknetwork.utils.neighbors import get_degrees


class DiffusionClassifier(BaseClassifier):
    """Node classification by heat diffusion.

    For each label, the temperature of a node corresponds to its probability to have this label.

    Parameters
    ----------
    n_iter : int
        Number of iterations of the diffusion (discrete time).
    centering : bool
        If ``True``, center the temperature of each label to its mean before classification (default).
    threshold : float
        Minimum difference of temperatures between the 2 top labels to classify a node (default = 0).
        If the difference of temperatures does not exceed this threshold, return -1 for this node (no label).

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

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> diffusion = DiffusionClassifier()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> seeds = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = diffusion.fit_predict(adjacency, seeds)
    >>> np.round(np.mean(labels_pred == labels_true), 2)
    0.97

    References
    ----------
    Zhu, X., Lafferty, J., & Rosenfeld, R. (2005). `Semi-supervised learning with graphs`
    (Doctoral dissertation, Carnegie Mellon University, language technologies institute, school of computer science).
    """
    def __init__(self, n_iter: int = 10, centering: bool = True, threshold: float = 0):
        super(DiffusionClassifier, self).__init__()

        if n_iter <= 0:
            raise ValueError('The number of iterations must be positive.')
        else:
            self.n_iter = n_iter
        self.centering = centering
        self.threshold = threshold
        self.bipartite = None

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            seeds: Optional[Union[dict, np.ndarray]] = None, seeds_row: Optional[Union[dict, np.ndarray]] = None,
            seeds_col: Optional[Union[dict, np.ndarray]] = None, force_bipartite: bool = False) \
            -> 'DiffusionClassifier':
        """Compute the solution to the Dirichlet problem (temperatures at equilibrium).

        Parameters
        ----------
        input_matrix :
            Adjacency matrix or biadjacency matrix of the graph.
        seeds :
            Labels of seed nodes (dictionary or vector of int). Negative values ignored.
        seeds_row, seeds_col :
            Labels of rows and columns for bipartite graphs. Negative values ignored.
        force_bipartite :
            If ``True``, consider the input matrix as a biadjacency matrix (default = ``False``).

        Returns
        -------
        self: :class:`DiffusionClassifier`
        """
        adjacency, seeds, self.bipartite = get_adjacency_seeds(input_matrix, force_bipartite=force_bipartite,
                                                               seeds=seeds, seeds_row=seeds_row, seeds_col=seeds_col)
        seeds = seeds.astype(int)
        if (seeds < 0).all():
            raise ValueError('At least one node must be given a label in ``seeds``.')
        temperatures = get_membership(seeds).toarray()
        temperatures_seeds = temperatures[seeds >= 0]
        n_labels = temperatures.shape[1]
        temperatures[seeds < 0] = 1 / n_labels
        diffusion = normalize(adjacency)
        for i in range(self.n_iter):
            temperatures = diffusion.dot(temperatures)
            temperatures[seeds >= 0] = temperatures_seeds

        self.membership_ = sparse.csr_matrix(temperatures)

        if self.centering:
            temperatures -= temperatures.mean(axis=0)

        labels = temperatures.argmax(axis=1)
        # set label -1 to nodes without temperature (no diffusion to them)
        labels[get_degrees(self.membership_) == 0] = -1

        if self.threshold >= 0:
            if n_labels > 2:
                top_temperatures = np.partition(-temperatures, 2, axis=1)[:, :2]
            else:
                top_temperatures = temperatures
            differences = np.abs(top_temperatures[:, 0] - top_temperatures[:, 1])
            labels[differences <= self.threshold] = -1

        self.labels_ = labels

        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self

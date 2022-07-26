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
from sknetwork.linalg import normalize
from sknetwork.utils.format import get_adjacency_seeds
from sknetwork.utils.membership import get_membership


class DiffusionClassifier(BaseClassifier):
    """Node classification by heat diffusion.

    For each label, the temperature of a node corresponds to its probability to have this label.

    Parameters
    ----------
    n_iter : int
        Number of iterations of the diffusion (discrete time).

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
    def __init__(self, n_iter: int = 10):
        super(DiffusionClassifier, self).__init__()

        if n_iter <= 0:
            raise ValueError('The number of iterations must be positive.')
        else:
            self.n_iter = n_iter
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
        temperatures = get_membership(seeds).toarray()
        temperatures_seeds = temperatures[seeds >= 0]
        temperatures[seeds < 0] = 1 / temperatures.shape[1]
        diffusion = normalize(adjacency)
        for i in range(self.n_iter):
            temperatures = diffusion.dot(temperatures)
            temperatures[seeds >= 0] = temperatures_seeds

        self.membership_ = sparse.csr_matrix(temperatures)
        self.labels_ = temperatures.argmax(axis=1)

        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self

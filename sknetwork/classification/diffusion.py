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
from sknetwork.path.distances import get_distances
from sknetwork.linalg.normalizer import normalize
from sknetwork.utils.format import get_adjacency_values
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
    scale : float
        Multiplicative factor applied to tempreatures before softmax (default = 5).
        Used only when centering is ``True``.

    Attributes
    ----------
    labels\_ : np.ndarray, shape (n_nodes,)
        Labels of nodes.
    probs\_ : sparse.csr_matrix, shape (n_nodes, n_labels)
        Probability distribution over labels.

    Example
    -------
    >>> from sknetwork.data import karate_club
    >>> diffusion = DiffusionClassifier()
    >>> graph = karate_club(metadata=True)
    >>> adjacency = graph.adjacency
    >>> labels_true = graph.labels
    >>> labels = {0: labels_true[0], 33: labels_true[33]}
    >>> labels_pred = diffusion.fit_predict(adjacency, labels)
    >>> float(round(np.mean(labels_pred == labels_true), 2))
    0.97

    References
    ----------
    Zhu, X., Lafferty, J., & Rosenfeld, R. (2005). `Semi-supervised learning with graphs`
    (Doctoral dissertation, Carnegie Mellon University, language technologies institute, school of computer science).
    """
    def __init__(self, n_iter: int = 10, centering: bool = True, scale: float = 5):
        super(DiffusionClassifier, self).__init__()

        if n_iter <= 0:
            raise ValueError('The number of iterations must be positive.')
        else:
            self.n_iter = n_iter
        self.centering = centering
        self.scale = scale

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray],
            labels: Optional[Union[dict, list, np.ndarray]] = None,
            labels_row: Optional[Union[dict, list, np.ndarray]] = None,
            labels_col: Optional[Union[dict, list, np.ndarray]] = None, force_bipartite: bool = False) \
            -> 'DiffusionClassifier':
        """Compute the solution to the Dirichlet problem (temperatures at equilibrium).

        Parameters
        ----------
        input_matrix : sparse.csr_matrix, np.ndarray
            Adjacency matrix or biadjacency matrix of the graph.
        labels : dict, np.ndarray
            Known labels (dictionary or vector of int). Negative values ignored.
        labels_row : dict, np.ndarray
            Labels of rows for bipartite graphs. Negative values ignored.
        labels_col : dict, np.ndarray
            Labels of columns for bipartite graphs. Negative values ignored.
        force_bipartite : bool
            If ``True``, consider the input matrix as a biadjacency matrix (default = ``False``).

        Returns
        -------
        self: :class:`DiffusionClassifier`
        """
        adjacency, values, self.bipartite = get_adjacency_values(input_matrix, force_bipartite=force_bipartite,
                                                                 values=labels,
                                                                 values_row=labels_row,
                                                                 values_col=labels_col)
        labels = values.astype(int)
        if (labels < 0).all():
            raise ValueError('At least one node must be given a non-negative label.')
        labels_reindex = labels.copy()
        labels_unique, inverse = np.unique(labels[labels >= 0], return_inverse=True)
        labels_reindex[labels >= 0] = inverse
        temperatures = get_membership(labels_reindex).toarray()
        temperatures_seeds = temperatures[labels >= 0]
        temperatures[labels < 0] = 0.5
        diffusion = normalize(adjacency)
        for i in range(self.n_iter):
            temperatures = diffusion.dot(temperatures)
            temperatures[labels >= 0] = temperatures_seeds
        if self.centering:
            temperatures -= temperatures.mean(axis=0)
        labels_ = labels_unique[temperatures.argmax(axis=1)]

        # softmax
        if self.centering:
            temperatures = np.exp(self.scale * temperatures)

        # set label -1 to nodes not reached by diffusion
        distances = get_distances(adjacency, source=np.flatnonzero(labels >= 0))
        labels_[distances < 0] = -1
        temperatures[distances < 0] = 0

        self.labels_ = labels_
        self.probs_ = sparse.csr_matrix(normalize(temperatures))
        self._split_vars(input_matrix.shape)

        return self

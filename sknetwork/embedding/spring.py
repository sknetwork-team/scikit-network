#!/usr/bin/env python3
# coding: utf-8
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.spectral import Spectral
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.utils.check import check_adjacency_vector, check_format, check_square, is_symmetric
from sknetwork.utils.format import directed2undirected
from sknetwork.linalg import normalize


class Spring(BaseEmbedding):
    """Spring layout for displaying small graphs.

    * Graphs
    * Digraphs

    Parameters
    ----------
    strength : float
        Intensity of the force that moves the nodes.
    n_iter : int
        Number of iterations to update positions.
    tol : float
        Minimum relative change in positions to continue updating.
    position_init : str
        How to initialize the layout. If 'spectral', use Spectral embedding in dimension 2,
        otherwise, use random initialization.

    Attributes
    ----------
    embedding_ : np.ndarray
        Layout in 2D.

    Example
    -------
    >>> from sknetwork.embedding import Spring
    >>> from sknetwork.data import karate_club
    >>> spring = Spring()
    >>> adjacency = karate_club()
    >>> embedding = spring.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    Notes
    -----
    Simple implementation designed to display small graphs in 2D.

    References
    ----------
    Fruchterman, T. M. J.,  Reingold, E. M. (1991).
    "Graph Drawing by Force-Directed Placement".
    Software – Practice & Experience.
    """
    def __init__(self, strength: float = None, n_iter: int = 50, tol: float = 1e-4, position_init: str = 'random'):
        super(Spring, self).__init__()
        self.strength = strength
        self.n_iter = n_iter
        self.tol = tol
        if position_init not in ['random', 'spectral']:
            raise ValueError('Unknown initial position, try "spectral" or "random".')
        else:
            self.position_init = position_init

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], position_init: Optional[np.ndarray] = None,
            n_iter: Optional[int] = None) -> 'Spring':
        """Compute layout.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph, treated as undirected.
        position_init : np.ndarray
            Custom initial positions of the nodes. Shape must be (n, 2).
            If ``None``, use the value of self.pos_init.
        n_iter : int
            Number of iterations to update positions.
            If ``None``, use the value of self.n_iter.

        Returns
        -------
        self: :class:`Spring`
        """
        adjacency = check_format(adjacency)
        check_square(adjacency)
        if not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)
        n = adjacency.shape[0]

        position = np.zeros((n, 2))
        if position_init is None:
            if self.position_init == 'random':
                position = np.random.randn(n, 2)
            elif self.position_init == 'spectral':
                position = Spectral(n_components=2, normalized=False).fit_transform(adjacency)
        elif isinstance(position_init, np.ndarray):
            if position_init.shape == (n, 2):
                position = position_init.copy()
            else:
                raise ValueError('Initial position has invalid shape.')
        else:
            raise TypeError('Initial position must be a numpy array.')

        if n_iter is None:
            n_iter = self.n_iter

        if self.strength is None:
            strength = np.sqrt((1 / n))
        else:
            strength = self.strength

        delta_x: float = position[:, 0].max() - position[:, 0].min()
        delta_y: float = position[:, 1].max() - position[:, 1].min()
        step_max: float = 0.1 * max(delta_x, delta_y)
        step: float = step_max / (n_iter + 1)

        delta = np.zeros((n, 2))
        for iteration in range(n_iter):
            delta *= 0
            for i in range(n):
                indices = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i+1]]
                data = adjacency.data[adjacency.indptr[i]:adjacency.indptr[i+1]]

                grad: np.ndarray = (position[i] - position)  # shape (n, 2)
                distance: np.ndarray = np.linalg.norm(grad, axis=1)  # shape (n,)
                distance = np.where(distance < 0.01, 0.01, distance)

                attraction = np.zeros(n)
                attraction[indices] += data * distance[indices] / strength

                repulsion = (strength / distance)**2

                delta[i]: np.ndarray = (grad * (repulsion - attraction)[:, np.newaxis]).sum(axis=0)  # shape (2,)
            length = np.linalg.norm(delta, axis=0)
            length = np.where(length < 0.01, 0.1, length)
            delta = delta * step_max / length
            position += delta
            step_max -= step
            err: float = np.linalg.norm(delta) / n
            if err < self.tol:
                break

        self.embedding_ = position
        return self

    def predict(self, adjacency_vectors: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict the embedding of new rows, defined by their adjacency vectors.

        Parameters
        ----------
        adjacency_vectors :
            Adjacency vectors of nodes.
            Array of shape (n_col,) (single vector) or (n_vectors, n_col)

        Returns
        -------
        embedding_vectors : np.ndarray
            Embedding of the nodes.
        """
        embedding = self.embedding_
        n = embedding.shape[0]
        if embedding is None:
            raise ValueError("This instance of Spring embedding is not fitted yet."
                             " Call 'fit' with appropriate arguments before using this method.")

        adjacency_vectors = check_adjacency_vector(adjacency_vectors, n)
        embedding_vectors = normalize(adjacency_vectors).dot(embedding)

        if embedding_vectors.shape[0] == 1:
            embedding_vectors = embedding_vectors.ravel()

        return embedding_vectors

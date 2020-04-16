#!/usr/bin/env python3
# coding: utf-8
"""
Created on Apr 2020
@author: Nathan de Lara <ndelara@enst.fr>
"""

from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding import GSVD
from sknetwork.embedding.base import BaseEmbedding
from sknetwork.utils.check import check_format, check_square, is_symmetric
from sknetwork.utils.format import directed2undirected


class FruchtermanReingold(BaseEmbedding):
    """Basic spring layout for displaying small graphs."""

    def __init__(self, strength: float = None, n_iter: int = 50, tol: float = 1e-4):
        super(FruchtermanReingold, self).__init__()
        self.strength = strength
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], pos_init: Union[np.ndarray, str] = 'spectral',
            n_iter: int = None) -> 'FruchtermanReingold':
        """Compute layout"""
        adjacency = check_format(adjacency)
        check_square(adjacency)
        if not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)
        n = adjacency.shape[0]

        if pos_init == 'spectral':
            pos = GSVD(n_components=2).fit_transform(adjacency)
        elif pos_init == 'random':
            pos = np.random.randn(n, 2)
        elif isinstance(pos_init, np.ndarray):
            if pos_init.shape == (n, 2):
                pos = pos_init.copy()
            else:
                raise ValueError('Initial position has invalid shape.')
        else:
            raise TypeError('Unknown initial position, try "spectral" or "random".')

        if n_iter is None:
            n_iter = self.n_iter

        if self.strength is None:
            strength = np.sqrt((1 / n))
        else:
            strength = self.strength

        step_max: float = 0.1 * max(max(pos[:, 0] - min(pos[:, 0]), max(pos[:, 1] - min(pos[:, 1]))))
        step: float = step_max / (n_iter + 1)
        delta = np.zeros((n, 2))
        for iteration in range(n_iter):
            delta *= 0
            for i in range(n):
                indices = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i+1]]
                data = adjacency.data[adjacency.indptr[i]:adjacency.indptr[i+1]]

                grad: np.ndarray = (pos[i] - pos).ravel()
                distance: np.ndarray = np.linalg.norm(grad, axis=1)
                distance = np.where(distance < 0.01, 0.01, distance)

                attraction = np.zeros(n)
                attraction[indices] *= data * distance[indices] / strength

                repulsion = (strength * distance)**2

                delta[i]: np.ndarray = (grad * (repulsion - attraction)).sum(axis=1)
            length = np.linalg.norm(delta, axis=0)
            length = np.where(length < 0.01, 0.1, length)
            delta = delta * step_max / length
            pos += delta
            step_max -= step
            err: float = np.linalg.norm(delta) / n
            if err < self.tol:
                break
                
        self.embedding_ = pos
        return self

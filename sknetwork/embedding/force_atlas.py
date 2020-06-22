#!/usr/bin/env python3
# coding: utf-8
"""
Created on Jun 2020
@author: Victor Manach <victor.manach@telecom-paris.fr>
@author: Rémi Jaylet <remi.jaylet@telecom-paris.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.utils import directed2undirected
from sknetwork.utils.check import check_format, is_symmetric, check_square


class forceAtlas2(BaseEmbedding):

    def __init__(self, n_iter: int = 50):
        super(forceAtlas2, self).__init__()

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], n_iter: Optional[int] = None) -> 'forceAtlas2':
        adjacency = check_format(adjacency)
        check_square(adjacency)
        if not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)
        n = adjacency.shape[0]

        position = np.zeros((n, 2))

        if n_iter is None:
            n_iter = self.n_iter

        delta_x: float = position[:, 0].max() - position[:, 0].min()  # max variation /x
        delta_y: float = position[:, 1].max() - position[:, 1].min()  # max variation /y
        step_max: float = max(delta_x, delta_y)
        step: float = step_max / (n_iter + 1)  # definition of step

        delta = np.zeros((n, 2))  # initialization of variation of position of nodes
        for iteration in range(n_iter):
            delta *= 0
            for i in range(n):
                indices = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
                data = adjacency.data[adjacency.indptr[i]:adjacency.indptr[i + 1]]

                grad: np.ndarray = (position[i] - position)  # shape (n, 2)
                distance: np.ndarray = np.linalg.norm(grad, axis=1)  # shape (n,)

                attraction = np.zeros(n)
                attraction[indices] += data * distance[indices]  # change attraction of connected nodes

                repulsion = (1 / distance)

                delta[i]: np.ndarray = (grad * (repulsion - attraction)[:, np.newaxis]).sum(axis=0)  # shape (2,)
            delta = delta * step_max
            position += delta
            step_max -= step

        self.embedding_ = position
        return self
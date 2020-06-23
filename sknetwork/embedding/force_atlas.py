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


class ForceAtlas2(BaseEmbedding):

    def __init__(self, n_iter: int = 50):
        super(ForceAtlas2, self).__init__()
        self.n_iter = n_iter

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], n_iter: Optional[int] = None) -> 'ForceAtlas2':
        """Compute layout.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph, treated as undirected.
        n_iter : int
            Number of iterations to update positions.
            If ``None``, use the value of self.n_iter.

        Returns
        -------
        self: :class:`ForceAtlas2`
        """
        adjacency = check_format(adjacency)
        check_square(adjacency)
        if not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)
        n = adjacency.shape[0]

        if n < 5000:
            tolerance = 0.1
        elif 5000 < n < 50000:
            tolerance = 1
        else:
            tolerance = 10

        position = np.random.randn(n, 2)

        if n_iter is None:
            n_iter = self.n_iter

        deg = adjacency.dot(np.ones(adjacency.shape[1])) + 1

        delta_x: float = position[:, 0].max() - position[:, 0].min()  # max variation /x
        delta_y: float = position[:, 1].max() - position[:, 1].min()  # max variation /y
        step_max: float = max(delta_x, delta_y)
        step: float = step_max / (n_iter + 1)  # definition of step

        delta = np.zeros((n, 2))  # initialization of variation of position of nodes
        forces_for_each_node = np.zeros(n)
        swing_vector = np.zeros(n)
        for iteration in range(n_iter):
            delta *= 0
            global_swing = 0
            global_traction = 0
            for i in range(n):
                indices = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]

                grad: np.ndarray = (position[i] - position)  # shape (n, 2)
                distance: np.ndarray = np.linalg.norm(grad, axis=1)  # shape (n,)
                distance = np.where(distance < 0.01, 0.01, distance)

                attraction = np.zeros(n)
                attraction[indices] = 100 * distance[indices]  # change attraction of connected nodes
                repulsion = 0.01 * (deg[i] + 1) * deg / distance

                force = (repulsion - attraction).sum()  # forces resultant applied on node i

                swing_node = np.abs(force - forces_for_each_node[i])  # force variation applied on node i
                swing_vector[i] = swing_node
                traction = np.abs(force + forces_for_each_node) / 2  # traction force applied on node i

                global_swing += (deg[i] + 1) * swing_node
                global_traction += (deg[i] + 1) * traction
                global_speed = tolerance * global_traction / global_swing  # computation of global variables

                node_speed = 1 * global_speed / (1 + global_speed * np.sqrt(swing_node))

                forces_for_each_node[i] = force  # force resultant update

                delta[i]: np.ndarray = node_speed * force
                #delta[i]: np.ndarray = (grad * (repulsion - attraction)[:, np.newaxis]).sum(axis=0)  # shape (2,)
            length = np.linalg.norm(delta, axis=0)
            length = np.where(length < 0.01, 0.1, length)
            delta = delta * step_max / length  # normalisation of distance between nodes
            position += delta  # calculating displacement and final position of points after iteration
            step_max -= step
            if swing_vector.all() < 0.01:
                break  # If the swing of all nodes is zero, then convergence is reached and we break.

        self.embedding_ = position
        return self

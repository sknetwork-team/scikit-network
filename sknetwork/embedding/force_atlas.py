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
from scipy.spatial import cKDTree

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.utils.check import check_format, is_symmetric, check_square
from sknetwork.utils.format import directed2undirected


class ForceAtlas(BaseEmbedding):
    """Force Atlas layout for displaying graphs.

    Parameters
    ----------
    n_components : int
        Dimension of the graph layout.
    n_iter : int
        Number of iterations to update positions.
        If ``None``, use the value of self.n_iter.
    approx_radius : float
        If a positive value is provided, only the nodes within this distance a given node are used to compute
        the repulsive force.
    lin_log : bool
        If ``True``, use lin-log mode.
    gravity_factor : float
        Gravity force scaling constant.
    repulsive_factor : float
        Repulsive force scaling constant.
    tolerance : float
        Tolerance defined in the swinging constant.
    speed : float
        Speed constant.
    speed_max : float
        Constant used to impose constrain on speed.

    Attributes
    ----------
    embedding\_ : np.ndarray, shape = (n_nodes, n_components)
        Embedding of the nodes.

    Example
    -------
    >>> from sknetwork.embedding.force_atlas import ForceAtlas
    >>> from sknetwork.data import karate_club
    >>> force_atlas = ForceAtlas()
    >>> adjacency = karate_club()
    >>> embedding = force_atlas.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    References
    ----------
    Jacomy M., Venturini T., Heymann S., Bastian M. (2014).
    `ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software.
    <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679>`_
    Plos One.
    """
    def __init__(self, n_components: int = 2, n_iter: int = 50, approx_radius: float = -1, lin_log: bool = False,
                 gravity_factor: float = 0.01, repulsive_factor: float = 0.1, tolerance: float = 0.1,
                 speed: float = 0.1, speed_max: float = 10):
        super(ForceAtlas, self).__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.approx_radius = approx_radius
        self.lin_log = lin_log
        self.gravity_factor = gravity_factor
        self.repulsive_factor = repulsive_factor
        self.tolerance = tolerance
        self.speed = speed
        self.speed_max = speed_max
        self.embedding_ = None

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], pos_init: Optional[np.ndarray] = None,
            n_iter: Optional[int] = None) -> 'ForceAtlas':
        """Compute layout.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph, treated as undirected.
        pos_init :
            Position to start with. Random if not provided.
        n_iter : int
            Number of iterations to update positions.
            If ``None``, use the value of self.n_iter.

        Returns
        -------
        self: :class:`ForceAtlas`
        """
        # verify the format of the adjacency matrix
        adjacency = check_format(adjacency)
        check_square(adjacency)
        if not is_symmetric(adjacency):
            adjacency = directed2undirected(adjacency)
        n = adjacency.shape[0]

        # setting of the tolerance according to the size of the graph
        if n < 5000:
            tolerance = 0.1
        elif 5000 <= n < 50000:  # pragma: no cover
            tolerance = 1
        else:  # pragma: no cover
            tolerance = 10

        if n_iter is None:
            n_iter = self.n_iter

        # initial position of the nodes of the graph
        if pos_init is None:
            position: np.ndarray = np.random.randn(n, self.n_components)
        else:
            if pos_init.shape != (n, self.n_components):
                raise ValueError('The initial position does not have valid dimensions.')
            else:
                position = pos_init
        # compute the vector with the degree of each node
        degree: np.ndarray = adjacency.dot(np.ones(adjacency.shape[1])) + 1

        # initialization of variation of position of nodes
        resultants = np.zeros(n)
        delta: np.ndarray = np.zeros((n, self.n_components))
        swing_vector: np.ndarray = np.zeros(n)
        global_speed = 1

        for iteration in range(n_iter):
            delta *= 0
            global_swing = 0
            global_traction = 0

            if self.approx_radius > 0:
                tree = cKDTree(position)
            else:
                tree = None

            for i in range(n):

                # attraction
                indices = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]
                attraction = position[i] - position[indices]

                if self.lin_log:
                    attraction = np.sign(attraction) * np.log(1 + np.abs(10 * attraction))
                attraction = attraction.sum(axis=0)

                # repulsion
                if tree is None:
                    neighbors = np.arange(n)
                else:
                    neighbors = tree.query_ball_point(position[i], self.approx_radius, p=2)

                grad: np.ndarray = (position[i] - position[neighbors])  # shape (n_neigh, n_components)
                distance: np.ndarray = np.linalg.norm(grad, axis=1)  # shape (n_neigh,)
                distance = np.where(distance < 0.01, 0.01, distance)
                repulsion = grad * (degree[neighbors] / distance)[:, np.newaxis]

                repulsion *= self.repulsive_factor * degree[i]
                repulsion = repulsion.sum(axis=0)

                # gravity
                gravity = self.gravity_factor * degree[i] * grad
                gravity = gravity.sum(axis=0)

                # forces resultant applied on node i for traction, swing and speed computation
                force = repulsion - attraction - gravity
                resultant_new: float = np.linalg.norm(force)
                resultant_old: float = resultants[i]

                swing_node: float = np.abs(resultant_new - resultant_old)  # force variation applied on node i
                swing_vector[i] = swing_node
                global_swing += (degree[i] + 1) * swing_node

                traction: float = np.abs(resultant_new + resultant_old) / 2  # traction force applied on node i
                global_traction += (degree[i] + 1) * traction

                node_speed = self.speed * global_speed / (1 + global_speed * np.sqrt(swing_node))
                if node_speed > self.speed_max / resultant_new:  # pragma: no cover
                    node_speed = self.speed_max / resultant_new

                delta[i]: np.ndarray = node_speed * force
                resultants[i] = resultant_new
                global_speed = tolerance * global_traction / global_swing

            position += delta  # calculating displacement and final position of points after iteration
            if (swing_vector < 1).all():
                break  # if the swing of all nodes is zero, then convergence is reached.

        self.embedding_ = position
        return self

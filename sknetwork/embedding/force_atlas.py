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
from sknetwork.embedding.building_tree import Cell
from sknetwork.utils import directed2undirected
from sknetwork.utils.check import check_format, is_symmetric, check_square


class ForceAtlas2(BaseEmbedding):
    """Force Atlas2 layout for displaying graphs.

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_components :
        Choose dimension of the graph layout
    n_iter : int
        Number of iterations to update positions.
        If ``None``, use the value of self.n_iter.
    barnes_hut :
        If True, compute repulsive forces with barnes_hut approximation
    lin_log :
        If True, activate an alternative formula for the attractive force
    gravity_factor :
        Gravity force scaling constant
    strong_gravity :
        If True, activate an alternative formula for the gravity force
    repulsive_factor :
        Repulsive force scaling constant
    weight_exponent :
        If different to 0, modify attraction force, the weights are raised to the power of 'exponent'
    no_hubs :
        If True, change the value of the attraction force
    tolerance :
        Tolerance defined in the swinging constant
    speed :
        Speed constant
    speed_max :
        Constant used to impose constrain on speed
    theta :
        Parameter used in barnes_hut algorithm

    Attributes
    ----------
    embedding_ : np.ndarray
        Layout in multiple dimension.

    Example
    -------
    >>> from sknetwork.embedding.force_atlas import ForceAtlas2
    >>> from sknetwork.data import karate_club
    >>> force_atlas = ForceAtlas2()
    >>> adjacency = karate_club()
    >>> embedding = force_atlas.fit_transform(adjacency)
    >>> embedding.shape
    (34, 2)

    Notes
    -----
    Implementation designed to display graphs in multiple dimension.

    References
    ----------
    Jacomy M., Venturini T., Heymann S., Bastian M. (2014).
    "ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software".
    Plos One.

    Barnes J, Hut P (1986).
    "A hierarchical o(n log n) force-calculation algorithm".
    Nature 324: 446–449.
    """

    def __init__(self, n_components: int = 2, n_iter: int = 50, barnes_hut: bool = True, lin_log: bool = False,
                 gravity_factor: float = 0.01, strong_gravity: bool = False, repulsive_factor: float = 0.01,
                 weight_exponent: int = 0, no_hubs: bool = False, tolerance: float = 0.1, speed: float = 0.1,
                 speed_max: float = 10, theta: float = 1.2):
        super(ForceAtlas2, self).__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.barnes_hut = barnes_hut
        self.lin_log = lin_log
        self.gravity_factor = gravity_factor
        self.strong_gravity = strong_gravity
        self.repulsive_factor = repulsive_factor
        self.weight_exponent = weight_exponent
        self.no_hubs = no_hubs
        self.tolerance = tolerance
        self.speed = speed
        self.speed_max = speed_max
        self.theta = theta

        if n_components > 2 and barnes_hut:
            raise ValueError('Barnes and Hut algorithm can only be used in 2D')

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], n_components: Optional[int] = None,
            n_iter: Optional[int] = None) -> 'ForceAtlas2':
        """Compute layout.

        Parameters
        ----------
        adjacency :
            Adjacency matrix of the graph, treated as undirected.
        n_components :
            Choose dimension of the graph layout
        n_iter : int
            Number of iterations to update positions.
            If ``None``, use the value of self.n_iter.

        Returns
        -------
        self: :class:`ForceAtlas2`
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
        elif 5000 <= n < 50000:
            tolerance = 1
        else:
            tolerance = 10

        if n_iter is None:
            n_iter = self.n_iter
        if n_components is None:
            n_components = self.n_components

        # initial position of the nodes of the graph
        position = np.random.randn(n, n_components)

        # compute the vector with the degree of each node
        degree = adjacency.dot(np.ones(adjacency.shape[1])) + 1

        delta = np.zeros((n, n_components))  # initialization of variation of position of nodes
        forces_for_each_node = np.zeros(n)
        swing_vector = np.zeros(n)
        global_speed = 1
        attraction = np.zeros(n)

        for iteration in range(n_iter):
            delta *= 0
            global_swing = 0
            global_traction = 0

            # tree construction
            root = Cell(position[:, 0].min(), position[:, 0].max(), position[:, 1].min(), position[:, 1].max())
            for i in range(n):
                root.add(position[i], degree[i])

            for i in range(n):
                repulsion = []
                attraction *= 0
                repulsion = []

                force = np.asarray([0, 0])  # shape (2,) force resultant applied on node i

                indices = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]

                grad: np.ndarray = position[i] - position  # shape (n, 2)
                distance: np.ndarray = np.linalg.norm(grad, axis=1)  # shape (n,)
                distance = np.where(distance < 0.01, 0.01, distance)
                attraction[indices] = grad[indices]  # change attraction of connected nodes

                if self.lin_log:
                    attraction = np.log(1 + attraction)
                if self.weight_exponent != 0:
                    data = adjacency.data[adjacency.indptr[i]:adjacency.indptr[i + 1]]
                    attraction = (data ** self.weight_exponent) * attraction
                if self.no_hubs:
                    attraction = attraction / (degree[i] + 1)

                repulsion = np.asarray(root.apply_force(position[i][0], position[i][1], degree[i], self.theta,
                                                        repulsion, self.repulsive_factor))

                distance_b = np.zeros((n, 2))

                for j in range(n):
                    distance_b[j] = [distance[j], distance[j]]

                gravity = self.gravity_factor * (degree[i] + 1) * grad / distance_b
                force = np.sum(attraction, axis=0) + np.sum(gravity, axis=0) + np.sum(repulsion,
                                                                                      axis=0)  # forces resultant applied on node i
                force_res = np.sum(force, axis=0)
                forces_for_each_node_res = np.sum(forces_for_each_node[i], axis=0)

                gravity = self.gravity_factor * (degree[i] + 1)
                if self.strong_gravity:
                    gravity = gravity * distance

                swing_node = np.abs(force_res - forces_for_each_node_res)  # force variation applied on node i

                swing_vector[i] = swing_node
                traction = np.abs(force_res + forces_for_each_node_res) / 2  # traction force applied on node i

                global_swing += (degree[i] + 1) * swing_node
                global_traction += (degree[i] + 1) * traction

                node_speed = self.speed * global_speed / (1 + global_speed * np.sqrt(swing_node))
                if node_speed > self.speed_max / abs(force_res):
                    node_speed = self.speed_max / abs(force_res)

                forces_for_each_node[i] = force  # force resultant update

                delta[i] = node_speed * force
                global_speed = tolerance * global_traction / global_swing

            position += delta  # calculating displacement and final position of points after iteration
            if (swing_vector < 0.01).all():
                break  # If the swing of all nodes is zero, then convergence is reached and we break.

        self.embedding_ = position
        return self

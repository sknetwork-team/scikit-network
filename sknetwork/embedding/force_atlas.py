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
    """Force Atlas2 layout for displaying graphs.

    * Graphs
    * Digraphs

    Parameters
    ----------
    n_components : int
        Choose dimension of the graph layout
    n_iter : int
        Number of iterations to update positions.
        If ``None``, use the value of self.n_iter.
    barnes_hut : bool
        If True, compute repulsive forces with barnes_hut approximation
    lin_log : bool
        If True, activate an alternative formula for the attractive force
    gravity_factor : float
        Gravity force scaling constant
    strong_gravity : bool
        If True, activate an alternative formula for the gravity force
    repulsive_factor : float
        Repulsive force scaling constant
    no_hubs : bool
        If True, change the value of the attraction force
    tolerance : float
        Tolerance defined in the swinging constant
    speed : float
        Speed constant
    speed_max : float
        Constant used to impose constrain on speed
    theta : float
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

    def __init__(self, n_components: int = 2, n_iter: int = 50, barnes_hut: bool = False, lin_log: bool = False,
                 gravity_factor: float = 0.01, strong_gravity: bool = False, repulsive_factor: float = 0.1,
                 no_hubs: bool = False, no_overlapping: bool = False, tolerance: float = 0.1, speed: float = 0.1,
                 speed_max: float = 10, theta: float = 1.2):
        super(ForceAtlas2, self).__init__()
        self.n_components = n_components
        self.n_iter = n_iter
        self.barnes_hut = barnes_hut
        self.lin_log = lin_log
        self.gravity_factor = gravity_factor
        self.strong_gravity = strong_gravity
        self.repulsive_factor = repulsive_factor
        self.no_hubs = no_hubs
        self.no_overlapping = no_overlapping
        self.tolerance = tolerance
        self.speed = speed
        self.speed_max = speed_max
        self.theta = theta

        if n_components > 2 and barnes_hut:
            raise ValueError('Barnes and Hut algorithm can only be used in 2D')

    def fit(self, adjacency: Union[sparse.csr_matrix, np.ndarray], n_components: Optional[int] = 2,
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

        if n_components > 2 and self.barnes_hut:
            raise ValueError('Barnes and Hut algorithm can only be used in 2D')

        # setting of the tolerance according to the size of the graph
        if n < 5000:
            tolerance = 0.1
        elif 5000 <= n < 50000:
            tolerance = 1
        else:
            tolerance = 10

        if n_iter is None:
            n_iter = self.n_iter

        # initial position of the nodes of the graph
        position: np.ndarray = np.random.randn(n, self.n_components)

        # compute the vector with the degree of each node
        degree: np.ndarray = adjacency.dot(np.ones(adjacency.shape[1])) + 1

        # initialization of variation of position of nodes
        delta: np.ndarray = np.zeros((n, self.n_components))
        forces_for_each_node: np.ndarray = np.zeros((n, self.n_components))
        swing_vector: np.ndarray = np.zeros(n)
        global_speed = 1
        attraction: np.ndarray = np.zeros((n, self.n_components))
        repulsion: np.ndarray = np.zeros(self.n_components)

        for iteration in range(n_iter):
            delta *= 0
            global_swing = 0
            global_traction = 0

            # tree construction
            if self.barnes_hut:
                root = Cell(position[:, 0].min(), position[:, 0].max(), position[:, 1].min(), position[:, 1].max())
                for i in range(n):
                    root.add(position[i], degree[i])

            for i in range(n):
                attraction *= 0
                repulsion *= 0

                indices = adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i + 1]]

                grad: np.ndarray = position[i] - position  # shape (n, d)
                distance: np.ndarray = np.linalg.norm(grad, axis=1)  # shape (n,)
                distance = np.where(distance < 0.01, 0.01, distance)

                attraction[indices] = grad[indices]  # shape (n, d) change attraction of connected nodes
                if self.lin_log:
                    attraction = np.sign(attraction) * np.log(1 + np.abs(10 * attraction))
                if self.no_hubs:
                    attraction = attraction / degree[i]
                if self.no_overlapping:
                    distance_border_to_border = distance - 1  # node's size = 1
                    if (np.abs(distance_border_to_border) > 0.1).all():
                        attraction[indices] = grad[indices]
                        distance = distance_border_to_border
                    else:
                        attraction *= 0
                        repulsion = np.sum((100 * degree[i] * grad * (degree / distance)[:, np.newaxis]
                         ), axis=0)

                if self.barnes_hut:
                    repulsion = np.asarray(root.apply_force(position[i], degree[i], self.theta, repulsion,
                                                            self.repulsive_factor))

                else:
                    repulsion = np.sum(
                        (self.repulsive_factor * degree[i] * grad * (degree / distance)[:, np.newaxis]
                         ), axis=0)
                gravity = self.gravity_factor * degree[i] * grad
                if self.strong_gravity:
                    gravity *= grad

                # forces resultant applied on node i for traction, swing and speed computation
                force: float = repulsion - np.sum(attraction, axis=0) - np.sum(gravity, axis=0)
                force_res: float = np.linalg.norm(force)
                forces_for_each_node_res: float = np.linalg.norm(forces_for_each_node[i])

                swing_node: float = np.abs(force_res - forces_for_each_node_res)  # force variation applied on node i
                swing_vector[i] = swing_node
                global_swing += (degree[i] + 1) * swing_node

                traction: float = np.abs(force_res + forces_for_each_node_res) / 2  # traction force applied on node i
                global_traction += (degree[i] + 1) * traction

                node_speed = self.speed * global_speed / (1 + global_speed * np.sqrt(swing_node))
                if node_speed > self.speed_max / abs(force_res):
                    node_speed = self.speed_max / abs(force_res)

                forces_for_each_node[i] = force  # force resultant update

                delta[i]: np.ndarray = node_speed * force

                global_speed = tolerance * global_traction / global_swing

            position += delta  # calculating displacement and final position of points after iteration
            if (swing_vector < 1).all():
                break  # if the swing of all nodes is zero, then convergence is reached and we break.

        self.embedding_ = position
        return self


class Cell:
    """Builds a quad-tree of cells used to apply the Barnes Hut approximation

    * Graphs

    Parameters
    ----------
    pos_min : np.ndarray
        A numpy array of the position if the min coordinates of the cell
    pos_max : np.ndarray
        A numpy array of the position of the max coordinates of the cell
    center : np.ndarray
        Coordinates of the center of mass of the cell
    children : np.ndarray
        Array that contains the children cells of the current cell
    n_particles : int
        Number of particles in the cell and in its children
    pos_particle : np.ndarray
        Coordinates of the particle in the cell if there is a particle in this cell
    particle_degree : int
        The degree of the particle, used to compute repulsion force

    References
    ----------
    Jacomy M., Venturini T., Heymann S., Bastian M. (2014).
    "ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software".
    Plos One.
    Barnes J., Hut P. (1986)
    "A hierarchical O(N log N) force-calculation algorithm".
    Nature 324: 446–449.
    Dierickx M., Portillo S. (2013).
    "N-Body Building, Working Out MPI Parallelization on Barnes-Hut Oct-Trees".
    CS205 class at Harvard's School and Engineering and Applied Sciences.
    """

    def __init__(self, x_min, x_max, y_min, y_max):  # position.shape (2, n_components)
        self.pos_min = np.asarray([x_min, y_min])
        self.pos_max = np.asarray([x_max, y_max])
        self.center = np.zeros(2)  # position of the center of mass of the cell
        self.children = None  # list of cells that are the children of the current cell
        self.n_particles = 0  # number of particles in the cells in its sub-cells
        self.pos_particle = None  # numpy array that contains the position of the particle if there is one in this cell
        self.particle_degree = None

    def is_in_cell(self, position: np.ndarray) -> bool:  # test if a particle is inside the cell's bounds
        return (self.pos_min <= position).all() and (position <= self.pos_max).all()

    def add(self, position: np.ndarray, degree: int):
        if not self.is_in_cell(position):
            return  # do nothing if the particle we want to add is not in the cell's bounds
        if self.n_particles > 0:
            if self.n_particles == 1:
                self.make_children()
                for child in self.children:
                    child.add(self.pos_particle, self.particle_degree)
                self.pos_particle = None
            for child in self.children:
                child.add(position, degree)
        else:
            self.pos_particle = position
            self.particle_degree = degree

        self.center = (self.n_particles * self.center + position) / float(self.n_particles + 1)
        self.n_particles += 1

    def make_children(self):  # create the 4 children of a cell
        pos_middle = (self.pos_min + self.pos_max) / 2

        child_1 = Cell(self.pos_min[0], pos_middle[0], pos_middle[1], self.pos_max[1])  # top left sub-cell
        child_2 = Cell(pos_middle[0], self.pos_max[0], pos_middle[1], self.pos_max[1])  # top right sub-cell
        child_3 = Cell(self.pos_min[0], pos_middle[0], self.pos_min[1], pos_middle[1])  # bottom left sub-cell
        child_4 = Cell(pos_middle[0], self.pos_max[0], self.pos_min[1], pos_middle[1])  # bottom right sub-cell

        self.children = np.asarray([child_1, child_2, child_3, child_4])

    def apply_force(self, pos_node, node_degree, theta, repulsion, repulsive_factor: float):
        if self.n_particles == 0:
            return
        cell_size = self.pos_max[0] - self.pos_min[0]
        grad: np.ndarray = pos_node - self.center
        if self.n_particles == 1:  # compute repulsion force between two nodes
            variation = self.pos_particle - pos_node
            distance = np.linalg.norm(grad, axis=0)
            if distance > 0:
                repulsion_force = repulsive_factor * node_degree * (self.n_particles + 1) * grad / distance
                repulsion += repulsion_force
        else:
            distance = np.linalg.norm(grad, axis=0)
            if distance * theta > cell_size:
                repulsion_force = repulsive_factor * node_degree * (self.n_particles + 1) * grad / distance
                repulsion += repulsion_force

            else:
                for sub_cell in self.children:
                    sub_cell.apply_force(pos_node, node_degree, theta, repulsion, repulsive_factor)
        return repulsion

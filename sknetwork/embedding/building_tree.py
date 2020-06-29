#!/usr/bin/env python3
# coding: utf-8

import numpy as np


class Cell:
    """Builds a quadtree of cells used to apply the Barnes Hut approximation

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

    Attributes
    ----------

    Example
    -------

    Notes
    -----

    References
    ----------
    Jacomy M., Venturini T., Heymann S., Bastian M. (2014).
    "ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software".
    Plos One.
    Barnes J., Hut P. (1986)
    "A hierarchical O(N log N) force-calculation algorithm".
    Nature 324: 446–449.
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
            distance = np.linalg.norm(variation, axis=0)
            if distance > 0:
                repulsion_force = repulsive_factor * node_degree * (self.n_particles + 1) * grad
                repulsion += repulsion_force
        else:
            distance = np.linalg.norm(grad, axis=0)
            if distance * theta > cell_size:
                repulsion_force = repulsive_factor * node_degree * (self.n_particles + 1) / grad
                repulsion += repulsion_force
            else:
                for sub_cell in self.children:
                    sub_cell.apply_force(pos_node, node_degree, theta, repulsion, repulsive_factor)
        return repulsion

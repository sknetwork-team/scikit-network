#!/usr/bin/env python3
# coding: utf-8

import numpy as np


class Cell:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.pos_min = np.asarray([x_min, y_min])
        self.pos_max = np.asarray([x_max, y_max])
        self.center = np.zeros(2)  # position of the center of mass of the cell
        self.children = None  # list of cells that are the children of the current cell
        self.nb_particles = 0  # number of particles in the cells in its sub-cells
        self.particle_pos = None  # numpy array that contains the position of the particle if there is one in this cell

    def is_in_cell(self, position: np.ndarray) -> bool:  # test if a particle is inside the cell's bounds
        if (self.pos_min <= position).all() and (position <= self.pos_max).all():
            return True
        else:
            return False

    def add(self, position: np.ndarray):
        if not self.is_in_cell(position):
            return  # do nothing if the particle we want to add is not in the cell's bounds
        if self.nb_particles > 0:
            if self.nb_particles == 1:
                self.make_children()
                for child in self.children:
                    child.add(self.particle_pos)
                self.particle_pos = None
            for child in self.children:
                child.add(position)
        else:
            self.particle_pos = position

        self.center = (self.nb_particles * self.center + position) / float(self.nb_particles + 1)
        self.nb_particles += 1

    def make_children(self):  # create the 4 children of a cell
        pos_middle = (self.pos_min + self.pos_max) / 2

        child_1 = Cell(self.pos_min[0], pos_middle[0], pos_middle[1], self.pos_max[1])  # top left sub-cell
        child_2 = Cell(pos_middle[0], self.pos_max[0], pos_middle[1], self.pos_max[1])  # top right sub-cell
        child_3 = Cell(self.pos_min[0], pos_middle[0], self.pos_min[1], pos_middle[1])  # bottom left sub-cell
        child_4 = Cell(pos_middle[0], self.pos_max[0], self.pos_min[1], pos_middle[1])  # bottom right sub-cell

        self.children = np.asarray([child_1, child_2, child_3, child_4])

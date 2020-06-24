#!/usr/bin/env python3
# coding: utf-8


class Particle:
    def __init__(self, x: float, y: float):
        self.x = x  # store the position of the particle in space with x and y
        self.y = y


class Cell:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.x_center, self.y_center = 0, 0  # position of the center of mass of the cell
        self.children = []  # list of cells that are the children of the current cell
        self.nb_particles = 0  # number of particles in the cells in its sub-cells
        self.particle = None  # object particle if there is a particle in this cell

    def is_in_cell(self, particle: Particle) -> bool:  # test if a particle is inside the cell's bounds
        if self.x_min < particle.x < self.x_max and self.y_min < particle.y < self.y_max:
            return True
        else:
            return False

    def add(self, particle: Particle):
        if not self.is_in_cell(particle):
            return  # do nothing if the particle we want to add is not in the cell's bounds

        if self.nb_particles > 0:
            if self.nb_particles == 1:
                self.make_children()
                for child in self.children:
                    child.add(self.particle)
                self.particle = None
            for child in self.children:
                child.add(particle)
        else:
            self.particle = particle

        self.x_center = (self.nb_particles * self.x_center + particle.x) / float(self.nb_particles + 1)
        self.y_center = (self.nb_particles * self.y_center + particle.y) / float(self.nb_particles + 1)
        self.nb_particles += 1

    def make_children(self):  # create the 4 children of a cell
        x_middle = (self.x_min + self.x_max) / 2
        y_middle = (self.y_min + self.y_max) / 2

        child_1 = Cell(self.x_min, x_middle, y_middle, self.y_max)  # top left sub-cell
        child_2 = Cell(x_middle, self.x_max, y_middle, self.y_max)  # top right sub-cell
        child_3 = Cell(self.x_min, x_middle, self.y_min, y_middle)  # bottom left sub-cell
        child_4 = Cell(x_middle, self.x_max, self.y_min, y_middle)  # bottom right sub-cell

        self.children = [child_1, child_2, child_3, child_4]

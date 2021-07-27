#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for eigenvalue solver."""

import unittest

import numpy as np

from sknetwork.data import miserables, karate_club
from sknetwork.linalg import LanczosEig, SparseLR


def eigenvector_err(matrix, eigenvectors, eigenvalues):
    """Approximation error for eigenvectors."""
    err = matrix.dot(eigenvectors) - eigenvectors * eigenvalues
    return np.linalg.norm(err)


# noinspection DuplicatedCode
class TestSolvers(unittest.TestCase):

    def setUp(self):
        """Load les Miserables and regularized version"""
        self.adjacency = miserables()
        self.random_state = np.random.RandomState(123)
        n = self.adjacency.shape[0]
        x = np.random.random(n)
        self.slr = SparseLR(self.adjacency, [(x, x)])

    def test_lanczos(self):
        solver = LanczosEig('LM')
        solver.fit(self.adjacency, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)
        self.assertAlmostEqual(eigenvector_err(self.adjacency, solver.eigenvectors_, solver.eigenvalues_), 0)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)
        self.assertAlmostEqual(eigenvector_err(self.slr, solver.eigenvectors_, solver.eigenvalues_), 0)

        adjacency = karate_club()
        solver = LanczosEig('SM')
        solver.fit(adjacency, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)
        self.assertAlmostEqual(eigenvector_err(adjacency, solver.eigenvectors_, solver.eigenvalues_), 0)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for svd"""

import unittest

import numpy as np

from sknetwork.linalg import LanczosSVD, HalkoSVD, SparseLR
from sknetwork.data import movie_actor


def svd_err(matrix, u, v, sigma):
    """Approximation error for singular vectors."""
    err = matrix.dot(v) - u * sigma
    return np.linalg.norm(err)


# noinspection DuplicatedCode
class TestSolvers(unittest.TestCase):

    def setUp(self):
        """Simple biadjacency for tests."""
        self.biadjacency = movie_actor()
        n_row, n_col = self.biadjacency.shape
        self.slr = SparseLR(self.biadjacency, [(np.random.rand(n_row), np.random.rand(n_col))])

    def test_lanczos(self):
        solver = LanczosSVD()
        solver.fit(self.biadjacency, 2)
        self.assertEqual(len(solver.singular_values_), 2)
        self.assertAlmostEqual(svd_err(self.biadjacency, solver.singular_vectors_left_, solver.singular_vectors_right_,
                                       solver.singular_values_), 0)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.singular_values_), 2)
        self.assertAlmostEqual(svd_err(self.slr, solver.singular_vectors_left_, solver.singular_vectors_right_,
                                       solver.singular_values_), 0)

    def test_halko(self):
        solver = HalkoSVD()
        solver.fit(self.biadjacency, 2)
        self.assertEqual(len(solver.singular_values_), 2)
        self.assertAlmostEqual(svd_err(self.biadjacency, solver.singular_vectors_left_, solver.singular_vectors_right_,
                                       solver.singular_values_), 0)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.singular_values_), 2)
        self.assertAlmostEqual(svd_err(self.slr, solver.singular_vectors_left_, solver.singular_vectors_right_,
                                       solver.singular_values_), 0)

    def test_compare_solvers(self):
        lanczos = LanczosSVD()
        halko = HalkoSVD()

        lanczos.fit(self.biadjacency, 2)
        halko.fit(self.biadjacency, 2)
        self.assertAlmostEqual(np.linalg.norm(lanczos.singular_values_ - halko.singular_values_), 0.)

        lanczos.fit(self.slr, 2)
        halko.fit(self.slr, 2)
        self.assertAlmostEqual(np.linalg.norm(lanczos.singular_values_ - halko.singular_values_), 0.)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for svd"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.linalg import LanczosSVD, HalkoSVD, SparseLR
from sknetwork.data import movie_actor


# noinspection PyMissingOrEmptyDocstring
def svd_err(matrix, u, v, sigma):
    err = matrix.dot(v) - u * sigma
    return np.linalg.norm(err)


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
class TestSolvers(unittest.TestCase):

    def setUp(self):
        self.biadjacency: sparse.csr_matrix = movie_actor()
        n, m = self.biadjacency.shape
        self.slr = SparseLR(self.biadjacency, [(np.random.rand(n), np.random.rand(m))])

    def test_lanczos(self):
        solver = LanczosSVD()
        solver.fit(self.biadjacency, 2)
        self.assertEqual(len(solver.singular_values_), 2)
        self.assertAlmostEqual(svd_err(self.biadjacency, solver.left_singular_vectors_, solver.right_singular_vectors_,
                                       solver.singular_values_), 0)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.singular_values_), 2)
        self.assertAlmostEqual(svd_err(self.slr, solver.left_singular_vectors_, solver.right_singular_vectors_,
                                       solver.singular_values_), 0)

    def test_halko(self):
        solver = HalkoSVD()
        solver.fit(self.biadjacency, 2)
        self.assertEqual(len(solver.singular_values_), 2)
        self.assertAlmostEqual(svd_err(self.biadjacency, solver.left_singular_vectors_, solver.right_singular_vectors_,
                                       solver.singular_values_), 0)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.singular_values_), 2)
        self.assertAlmostEqual(svd_err(self.slr, solver.left_singular_vectors_, solver.right_singular_vectors_,
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

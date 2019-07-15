#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for svd"""

import unittest
import numpy as np
from sknetwork.linalg import LanczosSVD, HalkoSVD, SparseLR
from sknetwork.toy_graphs import movie_actor


class TestSolvers(unittest.TestCase):

    def setUp(self):
        self.biadjacency = movie_actor()
        n, m = self.biadjacency.shape
        self.slr = SparseLR(self.biadjacency, [(np.ones(n), np.ones(m))])

    def test_lanczos(self):
        solver = LanczosSVD()
        solver.fit(self.biadjacency, 2)
        self.assertEqual(len(solver.singular_values_), 2)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.singular_values_), 2)

    def test_halko(self):
        solver = HalkoSVD()
        solver.fit(self.biadjacency, 2)
        self.assertEqual(len(solver.singular_values_), 2)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.singular_values_), 2)

    def compare_solvers(self):
        lanczos = LanczosSVD()
        halko = HalkoSVD()

        lanczos.fit(self.biadjacency, 2)
        halko.fit(self.biadjacency, 2)
        self.assertAlmostEqual(np.linalg.norm(lanczos.singular_values_ - halko.singular_values_), 0.)

        lanczos.fit(self.slr, 2)
        halko.fit(self.slr, 2)
        self.assertAlmostEqual(np.linalg.norm(lanczos.singular_values_ - halko.singular_values_), 0.)

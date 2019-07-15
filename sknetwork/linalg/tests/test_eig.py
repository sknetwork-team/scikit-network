#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for eig"""

import unittest
import numpy as np
from sknetwork.linalg import LanczosEig, HalkoEig, SparseLR
from sknetwork.toy_graphs import house


class TestSolvers(unittest.TestCase):

    def setUp(self):
        self.adjacency = house()
        n, m = self.adjacency.shape
        self.slr = SparseLR(self.adjacency, [(np.ones(n), np.ones(m))])

    def test_lanczos(self):
        solver = LanczosEig('LM')
        solver.fit(self.adjacency, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)

        solver = LanczosEig('SM')
        solver.fit(self.adjacency, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)

    def test_halko(self):
        solver = HalkoEig('LM')
        solver.fit(self.adjacency, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)

        solver.fit(self.slr, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)

        solver = HalkoEig('SM')
        solver.fit(self.adjacency, 2)
        self.assertEqual(len(solver.eigenvalues_), 2)

    def compare_solvers(self):
        lanczos = LanczosEig('LM')
        halko = HalkoEig('LM')

        lanczos.fit(self.adjacency, 2)
        halko.fit(self.adjacency, 2)
        self.assertAlmostEqual(np.linalg.norm(lanczos.eigenvalues_ - halko.eigenvalues_), 0.)

        lanczos.fit(self.slr, 2)
        halko.fit(self.slr, 2)
        self.assertAlmostEqual(np.linalg.norm(lanczos.eigenvalues_ - halko.eigenvalues_), 0.)

        lanczos = LanczosEig('SM')
        halko = HalkoEig('SM')

        lanczos.fit(self.adjacency, 2)
        halko.fit(self.adjacency, 2)
        self.assertAlmostEqual(np.linalg.norm(lanczos.eigenvalues_ - halko.eigenvalues_), 0.)

        lanczos.fit(self.slr, 2)
        halko.fit(self.slr, 2)
        self.assertAlmostEqual(np.linalg.norm(lanczos.eigenvalues_ - halko.eigenvalues_), 0.)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for diffusion.py"""

import unittest

from sknetwork.ranking import Diffusion, BiDiffusion, Dirichlet, BiDirichlet
from sknetwork.data.test_graphs import *


# noinspection DuplicatedCode
class TestDiffusion(unittest.TestCase):

    def setUp(self):
        self.algos = [Diffusion(), Dirichlet()]

    def test_n_iter(self):
        with self.assertRaises(ValueError):
            Diffusion(n_iter=-1)

    def test_single_node_graph(self):
        for algo in self.algos:
            algo.fit(sparse.identity(1, format='csr'), {0: 1})
            self.assertEqual(algo.scores_, [1])

    def test_range(self):
        for adjacency in [test_graph(), test_digraph()]:
            for algo in self.algos:
                scores = algo.fit_transform(adjacency, {0: 0, 1: 1, 2: 0.5})
                self.assertTrue(np.all(scores <= 1) and np.all(scores >= 0))

        biadjacency = test_bigraph()
        for algo in [BiDiffusion(), BiDirichlet()]:
            scores = algo.fit_transform(biadjacency, {0: 1})
            self.assertTrue(np.all(scores <= 1) and np.all(scores >= 0))
            scores = algo.fit_transform(biadjacency, {0: 0.1}, {1: 2}, 0.3)
            self.assertTrue(np.all(scores <= 2) and np.all(scores >= 0.1))

    def test_initial_state(self):
        for adjacency in [test_graph(), test_digraph()]:
            for algo in self.algos:
                scores = algo.fit_transform(adjacency, {0: 0, 1: 1, 2: 0.5}, 0.3)
                self.assertTrue(np.all(scores <= 1) and np.all(scores >= 0))

    def test_damping_factor(self):
        for adjacency in [test_graph(), test_digraph()]:
            for algo in [Diffusion(damping_factor=0.5), Dirichlet(damping_factor=0.5)]:
                scores = algo.fit_transform(adjacency, {0: 0, 1: 1, 2: 0.5})
                self.assertTrue(np.all(scores <= 1) and np.all(scores >= 0))

    def test_bicgstab(self):
        algo1 = Dirichlet(n_iter=0)
        algo2 = Dirichlet(n_iter=10)
        adjacency = test_digraph()
        scores1 = algo1.fit_transform(adjacency, {0: 1})
        scores2 = algo2.fit_transform(adjacency, {0: 1})
        self.assertAlmostEqual(np.linalg.norm(scores1 - scores2), 0)
        algo1 = Dirichlet(n_iter=0, damping_factor=0.5)
        algo2 = Dirichlet(n_iter=10, damping_factor=0.5)
        adjacency = test_digraph()
        scores1 = algo1.fit_transform(adjacency, {0: 1})
        scores2 = algo2.fit_transform(adjacency, {0: 1})
        self.assertAlmostEqual(np.linalg.norm(scores1 - scores2), 0)



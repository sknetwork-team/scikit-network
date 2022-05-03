#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for diffusion.py"""

import unittest

from sknetwork.data.test_graphs import *
from sknetwork.regression import Diffusion, Dirichlet


# noinspection DuplicatedCode
class TestDiffusion(unittest.TestCase):

    def setUp(self):
        self.algos = [Diffusion(), Dirichlet()]

    def test_no_iter(self):
        with self.assertRaises(ValueError):
            Diffusion(n_iter=-1)

    def test_single_node_graph(self):
        for algo in self.algos:
            algo.fit(sparse.identity(1, format='csr'), {0: 1})
            self.assertEqual(algo.values_, [1])

    def test_range(self):
        for adjacency in [test_graph(), test_digraph()]:
            for algo in self.algos:
                values = algo.fit_transform(adjacency, {0: 0, 1: 1, 2: 0.5})
                self.assertTrue(np.all(values <= 1) and np.all(values >= 0))

        biadjacency = test_bigraph()
        for algo in [Diffusion(), Dirichlet()]:
            values = algo.fit_transform(biadjacency, seeds_row={0: 1})
            self.assertTrue(np.all(values <= 1) and np.all(values >= 0))
            values = algo.fit_transform(biadjacency, seeds_row={0: 0.1}, seeds_col={1: 2}, init=0.3)
            self.assertTrue(np.all(values <= 2) and np.all(values >= 0.1))

    def test_initial_state(self):
        for adjacency in [test_graph(), test_digraph()]:
            for algo in self.algos:
                values = algo.fit_transform(adjacency, {0: 0, 1: 1, 2: 0.5}, 0.3)
                self.assertTrue(np.all(values <= 1) and np.all(values >= 0))

    def test_damping_factor(self):
        for adjacency in [test_graph(), test_digraph()]:
            for algo in [Diffusion(damping_factor=0.5), Dirichlet(damping_factor=0.5)]:
                values = algo.fit_transform(adjacency, {0: 0, 1: 1, 2: 0.5})
                self.assertTrue(np.all(values <= 1) and np.all(values >= 0))

    def test_n_iter(self):
        algo1 = Dirichlet(n_iter=0)
        algo2 = Dirichlet(n_iter=10)
        adjacency = test_digraph()
        values1 = algo1.fit_transform(adjacency, {0: 1})
        values2 = algo2.fit_transform(adjacency, {0: 1})
        self.assertAlmostEqual(np.linalg.norm(values1 - values2), 0)
        algo1 = Dirichlet(n_iter=0, damping_factor=0.5)
        algo2 = Dirichlet(n_iter=10, damping_factor=0.5)
        adjacency = test_digraph()
        values1 = algo1.fit_transform(adjacency, {0: 1})
        values2 = algo2.fit_transform(adjacency, {0: 1})
        self.assertAlmostEqual(np.linalg.norm(values1 - values2), 0)



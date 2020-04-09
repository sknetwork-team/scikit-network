#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for diffusion.py"""

import unittest

from sknetwork.ranking.diffusion import Diffusion, BiDiffusion
from sknetwork.data.test_graphs import *


# noinspection DuplicatedCode
class TestDiffusion(unittest.TestCase):

    def setUp(self):
        """Basic diffusion."""
        self.diffusion = Diffusion()

    def test_unknown_types(self):
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            self.diffusion.fit(sparse.identity(2, format='csr'), seeds='test')

    def test_single_node_graph(self):
        self.diffusion.fit(sparse.identity(1, format='csr'), {0: 1})
        self.assertEqual(self.diffusion.scores_, [1])

    def test_range(self):
        for adjacency in [test_graph(), test_digraph()]:
            score = self.diffusion.fit_transform(adjacency, {0: 0, 1: 1, 2: 0.5})
            self.assertTrue(np.all(score <= 1) and np.all(score >= 0))

        bidiffusion = BiDiffusion()
        biadjacency = test_bigraph()

        score = bidiffusion.fit_transform(biadjacency, {0: 1})
        self.assertTrue(np.all(score <= 1) and np.all(score >= 0))

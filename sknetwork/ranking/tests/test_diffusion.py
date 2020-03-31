#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for diffusion.py"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.ranking.diffusion import Diffusion, BiDiffusion
from sknetwork.data.test_graphs import *


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
class TestDiffusion(unittest.TestCase):

    def setUp(self):
        self.diffusion = Diffusion()
        self.naive_diff = Diffusion(n_iter=10)
        self.tol = 1e-6

    def test_unknown_types(self):
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            self.diffusion.fit(sparse.identity(2, format='csr'), personalization='test')

    def test_single_node_graph(self):
        self.diffusion.fit(sparse.identity(1, format='csr'), {0: 1})
        self.assertEqual(self.diffusion.scores_, [1])

    def test_undirected(self):
        adjacency = Simple().adjacency

        self.diffusion.fit(adjacency, {0: 0, 1: 1, 2: 0.5})
        score = self.diffusion.scores_
        self.assertTrue(np.all(score <= 1 + self.tol) and np.all(score >= 0 - self.tol))

        self.naive_diff.fit(adjacency, {0: 0, 1: 1, 2: 0.5})
        score = self.diffusion.scores_
        self.assertTrue(np.all(score <= 1 + self.tol) and np.all(score >= 0 - self.tol))

    def test_directed(self):
        adjacency = DiSimple().adjacency
        self.diffusion.fit(adjacency, {0: 0, 1: 1, 2: 0.5})
        score = self.diffusion.scores_
        self.assertTrue(np.all(score <= 1 + self.tol) and np.all(score >= 0 - self.tol))

    def test_bidiffusion(self):
        bidiffusion = BiDiffusion()
        biadjacency = BiSimple().biadjacency
        bidiffusion.fit(biadjacency, {0: 1})
        score = bidiffusion.scores_
        self.assertTrue(np.all(score <= 1) and np.all(score >= 0))

        naive_bidiff = BiDiffusion(n_iter=10)
        score = naive_bidiff.fit_transform(biadjacency, {0: 1})
        self.assertTrue(np.all(score <= 1) and np.all(score >= 0))

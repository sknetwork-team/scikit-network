#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for diffusion.py"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.ranking.diffusion import Diffusion, BiDiffusion
from sknetwork.toy_graphs import karate_club, painters, star_wars_villains


class TestDiffusion(unittest.TestCase):

    def setUp(self):
        self.diffusion = Diffusion()
        self.bidiffusion = BiDiffusion()
        self.karate_club = karate_club()
        self.painters = painters(return_labels=False)
        self.star_wars: sparse.csr_matrix = star_wars_villains(return_labels=False)
        self.tol = 1e-6

    def test_unknown_types(self):
        with self.assertRaises(ValueError):
            self.diffusion.fit(sparse.identity(2, format='csr'), personalization='test')

    def test_single_node_graph(self):
        self.diffusion.fit(sparse.identity(1, format='csr'), {0: 1})
        self.assertEqual(self.diffusion.score_, [1])

    def test_undirected(self):
        adjacency: sparse.csr_matrix = karate_club()
        self.diffusion.fit(adjacency, {0: 0, 1: 1, 2: -1})
        score = self.diffusion.score_
        self.assertTrue(np.all(score <= 1 + self.tol) and np.all(score >= -1 - self.tol))

    def test_directed(self):
        adjacency: sparse.csr_matrix = self.painters
        self.diffusion.fit(adjacency, {0: 0, 1: 1, 2: -1})
        score = self.diffusion.score_
        self.assertTrue(np.all(score <= 1 + self.tol) and np.all(score >= -1 - self.tol))

    def test_bidiff(self):
        self.bidiffusion.fit(self.star_wars, {0: 1})
        score = self.bidiffusion.score_
        self.assertTrue(np.all(score <= 1) and np.all(score >= 0))

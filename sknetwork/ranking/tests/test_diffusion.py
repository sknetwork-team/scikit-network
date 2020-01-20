#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for diffusion.py"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.ranking.diffusion import Diffusion, BiDiffusion
from sknetwork.data import karate_club, painters, star_wars_villains


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
        self.karate_club = karate_club()
        adjacency: sparse.csr_matrix = karate_club()

        self.diffusion.fit(adjacency, {0: 0, 1: 1, 2: 0.5})
        score = self.diffusion.scores_
        self.assertTrue(np.all(score <= 1 + self.tol) and np.all(score >= 0 - self.tol))

        self.naive_diff.fit(adjacency, {0: 0, 1: 1, 2: 0.5})
        score = self.diffusion.scores_
        self.assertTrue(np.all(score <= 1 + self.tol) and np.all(score >= 0 - self.tol))

    def test_directed(self):
        self.painters = painters(return_labels=False)
        adjacency: sparse.csr_matrix = self.painters
        self.diffusion.fit(adjacency, {0: 0, 1: 1, 2: 0.5})
        score = self.diffusion.scores_
        self.assertTrue(np.all(score <= 1 + self.tol) and np.all(score >= 0 - self.tol))

    def test_bidiffusion(self):
        bidiffusion = BiDiffusion()
        self.star_wars: sparse.csr_matrix = star_wars_villains(return_labels=False)
        bidiffusion.fit(self.star_wars, {0: 1})
        score = bidiffusion.scores_
        self.assertTrue(np.all(score <= 1) and np.all(score >= 0))

        naive_bidiff = BiDiffusion(n_iter=10)
        score = naive_bidiff.fit_transform(self.star_wars, {0: 1})
        self.assertTrue(np.all(score <= 1) and np.all(score >= 0))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for diffusion.py"""


import unittest
import numpy as np
from scipy.sparse import identity
from sknetwork.toy_graphs import karate_club, painters
from sknetwork.ranking.diffusion import Diffusion


class TestDiffusion(unittest.TestCase):

    def setUp(self):
        self.diffusion = Diffusion()
        self.karate_club = karate_club()
        self.painters = painters(return_labels=False)

    def test_unknown_types(self):
        with self.assertRaises(ValueError):
            self.diffusion.fit(identity(2, format='csr'), personalization='test')

    def test_single_node_graph(self):
        self.diffusion.fit(identity(1, format='csr'), {0: 1})
        self.assertEqual(self.diffusion.score_, [1])

    def test_undirected(self):
        adjacency = karate_club()
        self.diffusion.fit(adjacency, {0: 0, 1: 1, 2: -1})
        score = self.diffusion.score_
        self.assertTrue(np.all(score <= 1) and np.all(score >= -1))

    def test_directed(self):
        adjacency = self.painters
        self.diffusion.fit(adjacency, {0: 0, 1: 1, 2: -1})
        score = self.diffusion.score_
        self.assertTrue(np.all(score <= 1) and np.all(score >= -1))

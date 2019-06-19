#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""


import unittest
import numpy as np
from sknetwork.toy_graphs import rock_paper_scissors_graph
from sknetwork.ranking.pagerank import pagerank


class TestPagerank(unittest.TestCase):

    def setUp(self):
        self.adjacency = rock_paper_scissors_graph()

    def test_pagerank(self):
        ranking = pagerank(self.adjacency)
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0.)
        ranking = pagerank(self.adjacency, damping_factor=0.)
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0.)
        ranking = pagerank(self.adjacency, personalization=np.array([1, 0, 0]))
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0.)

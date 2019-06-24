#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""


import unittest
import numpy as np
from sknetwork.toy_graphs import rock_paper_scissors_graph
from sknetwork.ranking.pagerank import PageRank


class TestPageRank(unittest.TestCase):

    def setUp(self):
        self.adjacency = rock_paper_scissors_graph()
        self.spectral_ranking = PageRank(method='spectral')
        self.no_damping_spectral_ranking = PageRank(damping_factor=0., method='spectral')
        self.diter_ranking = PageRank()
        self.no_damping_diter_ranking = PageRank(damping_factor=0.)

    def test_pagerank(self):
        ranking = self.spectral_ranking.fit(self.adjacency).ranking_
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0.)
        ranking = self.no_damping_spectral_ranking.fit(self.adjacency).ranking_
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0.)
        ranking = self.spectral_ranking.fit(self.adjacency, personalization=np.array([1, 0, 0])).ranking_
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0.)
        ranking = self.diter_ranking.fit(self.adjacency).ranking_
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0., places=2)
        ranking = self.no_damping_diter_ranking.fit(self.adjacency).ranking_
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0., places=2)
        ranking = self.diter_ranking.fit(self.adjacency, personalization=np.array([1, 0, 0])).ranking_
        self.assertAlmostEqual(np.linalg.norm(ranking - np.ones(3) / 3), 0., places=1)

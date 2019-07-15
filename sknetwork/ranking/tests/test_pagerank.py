#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for pagerank.py"""


import unittest
import numpy as np
from sknetwork.toy_graphs import rock_paper_scissors
from sknetwork.ranking.pagerank import PageRank


class TestPageRank(unittest.TestCase):

    def setUp(self):
        self.adjacency = rock_paper_scissors()
        self.spectral_ranking = PageRank(method='spectral')
        self.no_damping_spectral_ranking = PageRank(damping_factor=0., method='spectral')
        self.diter_ranking = PageRank()
        self.no_damping_diter_ranking = PageRank(damping_factor=0.)
        self.parallel_ranking = PageRank(parallel=True)

    def test_pagerank(self):
        score = self.spectral_ranking.fit(self.adjacency).score_
        self.assertAlmostEqual(np.linalg.norm(score - np.ones(3) / 3), 0.)
        score = self.no_damping_spectral_ranking.fit(self.adjacency).score_
        self.assertAlmostEqual(np.linalg.norm(score - np.ones(3) / 3), 0.)
        score = self.spectral_ranking.fit(self.adjacency, personalization=np.array([1, 0, 0])).score_
        self.assertAlmostEqual(np.linalg.norm(score - np.ones(3) / 3), 0.)
        score = self.diter_ranking.fit(self.adjacency).score_
        self.assertAlmostEqual(np.linalg.norm(score - np.ones(3) / 3), 0., places=2)
        score = self.no_damping_diter_ranking.fit(self.adjacency).score_
        self.assertAlmostEqual(np.linalg.norm(score - np.ones(3) / 3), 0., places=2)
        score = self.diter_ranking.fit(self.adjacency, personalization=np.array([1, 0, 0])).score_
        self.assertAlmostEqual(np.linalg.norm(score - np.ones(3) / 3), 0., places=1)

    def test_parallelize(self):
        score = self.parallel_ranking.fit(self.adjacency).score_
        self.assertIsNotNone(score)

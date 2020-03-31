#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for PageRankClassifier"""

import unittest

from sknetwork.classification import PageRankClassifier, BiPageRankClassifier
from sknetwork.data.basic import *


# noinspection DuplicatedCode
class TestPageRankClassifier(unittest.TestCase):

    def test_undirected(self):
        adjacency = Small().adjacency
        adj_array_seeds = -np.ones(adjacency.shape[0])
        adj_array_seeds[:2] = np.arange(2)
        adj_dict_seeds = {0: 0, 1: 1}

        mr = PageRankClassifier(solver='lsqr')
        labels1 = mr.fit_transform(adjacency, adj_array_seeds)
        labels2 = mr.fit_transform(adjacency, adj_dict_seeds)
        self.assertTrue(np.allclose(labels1, labels2))
        self.assertEqual(labels2.shape[0], adjacency.shape[0])

    def test_bipartite(self):
        biadjacency = BiSmall().biadjacency
        biadj_array_seeds = -np.ones(biadjacency.shape[0])
        biadj_array_seeds[:2] = np.arange(2)
        biadj_dict_seeds = {0: 0, 1: 1}

        bmr = BiPageRankClassifier()
        labels1 = bmr.fit_transform(biadjacency, biadj_array_seeds)
        labels2 = bmr.fit_transform(biadjacency, biadj_dict_seeds)
        self.assertTrue(np.allclose(labels1, labels2))
        self.assertEqual(labels2.shape[0], biadjacency.shape[0])

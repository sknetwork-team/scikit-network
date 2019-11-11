#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for MultiRank"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.soft_cluster import MultiRank, BiMultiRank
from sknetwork.toy_graphs import painters, movie_actor


class TestMultiRank(unittest.TestCase):

    def setUp(self):
        self.adjacency: sparse.csr_matrix = painters()
        adj_array_seeds = -np.ones(self.adjacency.shape[0])
        adj_array_seeds[:2] = np.arange(2)
        self.adj_array_seeds = adj_array_seeds
        self.adj_dict_seeds = {0: 0, 1: 1}

        self.biadjacency: sparse.csr_matrix = movie_actor()
        badj_array_seeds = -np.ones(self.biadjacency.shape[0])
        badj_array_seeds[:2] = np.arange(2)
        self.badj_array_seeds = badj_array_seeds
        self.badj_dict_seeds = {0: 0, 1: 1}

    def test_multirank(self):
        mr = MultiRank(sparse_output=False)
        mr.fit(self.adjacency, self.adj_array_seeds)
        membership1 = mr.membership_.copy()
        mr.fit(self.adjacency, self.adj_dict_seeds)
        membership2 = mr.membership_.copy()
        self.assertTrue(np.allclose(membership1, membership2))
        self.assertEqual(membership2.shape, (self.adjacency.shape[0], 2))

    def test_bimultirank(self):
        bmr = BiMultiRank(sparse_output=False)
        bmr.fit(self.biadjacency, self.badj_array_seeds)
        membership1 = bmr.membership_.copy()
        bmr.fit(self.biadjacency, self.badj_dict_seeds)
        membership2 = bmr.membership_.copy()
        self.assertTrue(np.allclose(membership1, membership2))
        self.assertEqual(membership2.shape, (self.biadjacency.shape[0], 2))

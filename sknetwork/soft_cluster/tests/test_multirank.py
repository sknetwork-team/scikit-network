#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for MultiRank"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.soft_cluster import MultiRank, BiMultiRank
from sknetwork.data import painters, movie_actor


# noinspection DuplicatedCode
class TestMultiRank(unittest.TestCase):

    def test_multirank(self):
        self.adjacency: sparse.csr_matrix = painters()
        adj_array_seeds = -np.ones(self.adjacency.shape[0])
        adj_array_seeds[:2] = np.arange(2)
        self.adj_array_seeds = adj_array_seeds
        self.adj_dict_seeds = {0: 0, 1: 1}

        mr = MultiRank(sparse_output=False)
        mr.fit(self.adjacency, self.adj_array_seeds)
        membership1 = mr.membership_.copy()
        mr.fit(self.adjacency, self.adj_dict_seeds)
        membership2 = mr.membership_.copy()
        self.assertTrue(np.allclose(membership1, membership2))
        self.assertEqual(membership2.shape, (self.adjacency.shape[0], 2))

    def test_bimultirank(self):
        self.biadjacency: sparse.csr_matrix = movie_actor()
        biadj_array_seeds = -np.ones(self.biadjacency.shape[0])
        biadj_array_seeds[:2] = np.arange(2)
        self.biadj_array_seeds = biadj_array_seeds
        self.biadj_dict_seeds = {0: 0, 1: 1}

        bmr = BiMultiRank(sparse_output=False)
        bmr.fit(self.biadjacency, self.biadj_array_seeds)
        membership1 = bmr.membership_.copy()
        bmr.fit(self.biadjacency, self.biadj_dict_seeds)
        membership2 = bmr.membership_.copy()
        self.assertTrue(np.allclose(membership1, membership2))
        self.assertEqual(membership2.shape, (self.biadjacency.shape[0], 2))

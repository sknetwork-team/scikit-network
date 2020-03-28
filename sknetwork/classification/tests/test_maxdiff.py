#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for DiffusionClassifier"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.classification import DiffusionClassifier, BiDiffusionClassifier
from sknetwork.data import painters, movie_actor


# noinspection DuplicatedCode
class TestMaxDiff(unittest.TestCase):

    def test_maxrdiff(self):
        adjacency: sparse.csr_matrix = painters()
        adj_array_seeds = -np.ones(adjacency.shape[0])
        adj_array_seeds[:2] = np.arange(2)
        adj_dict_seeds = {0: 0, 1: 1}

        md = DiffusionClassifier()
        labels1 = md.fit_transform(adjacency, adj_array_seeds)
        labels2 = md.fit_transform(adjacency, adj_dict_seeds)
        self.assertTrue(np.allclose(labels1, labels2))
        self.assertEqual(labels2.shape[0], adjacency.shape[0])

    def test_bimaxdiff(self):
        biadjacency: sparse.csr_matrix = movie_actor()
        biadj_array_seeds = -np.ones(biadjacency.shape[0])
        biadj_array_seeds[:2] = np.arange(2)
        biadj_dict_seeds = {0: 0, 1: 1}

        bmd = BiDiffusionClassifier()
        labels1 = bmd.fit_transform(biadjacency, biadj_array_seeds)
        labels2 = bmd.fit_transform(biadjacency, biadj_dict_seeds)
        self.assertTrue(np.allclose(labels1, labels2))
        self.assertEqual(labels2.shape[0], biadjacency.shape[0])

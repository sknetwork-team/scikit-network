#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Louvain"""
import unittest

import numpy as np

from sknetwork.data import house, edgelist2adjacency
from sknetwork.linalg.operators import RegularizedAdjacency
from sknetwork.linalg.ppr_solver import get_pagerank
from sknetwork.utils.check import is_proba_array


class TestPPR(unittest.TestCase):

    def test_diteration(self):
        # test convergence by tolerance
        adjacency = house()
        seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
        pr = get_pagerank(adjacency, damping_factor=0.85, n_iter=100, tol=10, solver='diteration', seeds=seeds)
        self.assertTrue(is_proba_array(pr))

        # test graph with some null out-degree
        adjacency = edgelist2adjacency([(0, 1)])
        seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
        pr = get_pagerank(adjacency, damping_factor=0.85, n_iter=100, tol=10, solver='diteration', seeds=seeds)
        self.assertTrue(is_proba_array(pr))

        # test invalid entry
        adjacency = RegularizedAdjacency(house(), 0.1)
        seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
        with self.assertRaises(ValueError):
            get_pagerank(adjacency, damping_factor=0.85, n_iter=100, tol=10, solver='diteration', seeds=seeds)

    def test_piteration(self):
        # test on SparseLR matrix
        adjacency = RegularizedAdjacency(house(), 0.1)
        seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
        pr = get_pagerank(adjacency, damping_factor=0.85, n_iter=100, tol=10, solver='piteration', seeds=seeds)
        self.assertTrue(is_proba_array(pr))

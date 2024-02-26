#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for d-iteration"""
import unittest

import numpy as np

from sknetwork.data import house, karate_club
from sknetwork.data.parse import from_edge_list
from sknetwork.data.test_graphs import *
from sknetwork.linalg.operators import Regularizer
from sknetwork.linalg.ppr_solver import get_pagerank
from sknetwork.utils.check import is_proba_array


class TestPPR(unittest.TestCase):

    def test_diteration(self):
        # test convergence by tolerance
        for adjacency in [house(), test_graph(), test_digraph()]:
            seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
            pr = get_pagerank(adjacency, damping_factor=0.85, n_iter=100, tol=10, solver='diteration', seeds=seeds)
            self.assertTrue(is_proba_array(pr))

        # test graph with some null out-degree
        adjacency = from_edge_list([(0, 1)])
        seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
        pr = get_pagerank(adjacency, damping_factor=0.85, n_iter=100, tol=10, solver='diteration', seeds=seeds)
        self.assertTrue(is_proba_array(pr))

        # test invalid entry
        adjacency = Regularizer(house(), 0.1)
        seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
        with self.assertRaises(ValueError):
            get_pagerank(adjacency, damping_factor=0.85, n_iter=100, tol=10, solver='diteration', seeds=seeds)

    def test_push(self):
        # test convergence by tolerance
        adjacency = karate_club()
        seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
        pr = get_pagerank(adjacency, damping_factor=0.85,
                          n_iter=100, tol=1e-1, solver='push', seeds=seeds)
        self.assertTrue(is_proba_array(pr))

    def test_piteration(self):
        # test on SparseLR matrix
        adjacency = Regularizer(house(), 0.1)
        seeds = np.ones(adjacency.shape[0]) / adjacency.shape[0]
        pr = get_pagerank(adjacency, damping_factor=0.85, n_iter=100, tol=10, solver='piteration', seeds=seeds)
        self.assertTrue(is_proba_array(pr))

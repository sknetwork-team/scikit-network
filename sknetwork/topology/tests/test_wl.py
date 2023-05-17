#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Weisfeiler-Lehman"""
import unittest

import numpy as np

from sknetwork.data import house, bow_tie, linear_graph
from sknetwork.data.test_graphs import *
from sknetwork.topology import color_weisfeiler_lehman, are_isomorphic


class TestWLKernel(unittest.TestCase):

    def test_isomorphism(self):
        ref = house()
        n = ref.shape[0]

        adjacency = house()
        reorder = list(range(n))
        np.random.shuffle(reorder)
        adjacency = adjacency[reorder][:, reorder]
        self.assertTrue(are_isomorphic(ref, adjacency))

        adjacency = bow_tie()
        self.assertFalse(are_isomorphic(ref, adjacency))

        adjacency = linear_graph(n)
        self.assertFalse(are_isomorphic(ref, adjacency))

        adjacency = linear_graph(n + 1)
        self.assertFalse(are_isomorphic(ref, adjacency))


class TestWLColoring(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        labels = color_weisfeiler_lehman(adjacency)
        self.assertTrue((labels == np.zeros(10)).all())

    def test_cliques(self):
        adjacency = test_clique()
        labels = color_weisfeiler_lehman(adjacency)
        self.assertTrue((labels == np.zeros(10)).all())

    def test_house(self):
        adjacency = house()
        labels = color_weisfeiler_lehman(adjacency)
        self.assertTrue((labels == np.array([0, 2, 1, 1, 2])).all())

    def test_bow_tie(self):
        adjacency = bow_tie()
        labels = color_weisfeiler_lehman(adjacency)
        self.assertTrue((labels == np.array([1, 0, 0, 0, 0])).all())

    def test_iso(self):
        adjacency = house()
        n = adjacency.indptr.shape[0] - 1
        reorder = list(range(n))
        np.random.shuffle(reorder)
        adjacency2 = adjacency[reorder][:, reorder]
        l1 = color_weisfeiler_lehman(adjacency)
        l2 = color_weisfeiler_lehman(adjacency2)
        l1.sort()
        l2.sort()
        self.assertTrue((l1 == l2).all())

    def test_early_stop(self):
        adjacency = house()
        labels = color_weisfeiler_lehman(adjacency, max_iter=1)
        self.assertTrue((labels == np.array([0, 1, 0, 0, 1])).all())

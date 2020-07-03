#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Weisfeiler-Lehman coloring"""
import unittest
from sknetwork.topology import WLColoring
from sknetwork.data.test_graphs import *
from sknetwork.data import house, bow_tie


class TestWLColoring(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        labels = WLColoring().fit_transform(adjacency)
        self.assertTrue((labels == np.zeros(10)).all())

    def test_cliques(self):
        adjacency = test_graph_clique()
        labels = WLColoring().fit_transform(adjacency)
        self.assertTrue((labels == np.zeros(10)).all())

    def test_house(self):
        adjacency = house()
        labels = WLColoring().fit_transform(adjacency)
        self.assertTrue((labels == np.array([2, 0, 1, 1, 0])).all())

    def test_bow_tie(self):
        adjacency = bow_tie()
        labels = WLColoring().fit_transform(adjacency)
        self.assertTrue((labels == np.array([0, 1, 1, 1, 1])).all())

    def test_iso(self):
        adjacency = house()
        n = adjacency.indptr.shape[0] - 1
        reorder = list(range(n))
        np.random.shuffle(reorder)
        adjacency2 = adjacency[reorder][:, reorder]
        l1 = WLColoring().fit_transform(adjacency)
        l2 = WLColoring().fit_transform(adjacency2)
        l1.sort()
        l2.sort()
        self.assertTrue((l1 == l2).all())

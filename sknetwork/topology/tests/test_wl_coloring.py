#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Weisfeiler-Lehman coloring"""
import unittest

from sknetwork.data import house, bow_tie
from sknetwork.data.test_graphs import *
from sknetwork.topology import WeisfeilerLehman


class TestWLColoring(unittest.TestCase):

    def test_empty(self):
        adjacency = test_graph_empty()
        labels = WeisfeilerLehman().fit_transform(adjacency)
        self.assertTrue((labels == np.zeros(10)).all())

    def test_cliques(self):
        adjacency = test_graph_clique()
        labels = WeisfeilerLehman().fit_transform(adjacency)
        self.assertTrue((labels == np.zeros(10)).all())

    def test_house(self):
        adjacency = house()
        labels = WeisfeilerLehman().fit_transform(adjacency)
        self.assertTrue((labels == np.array([0, 2, 1, 1, 2])).all())

    def test_bow_tie(self):
        adjacency = bow_tie()
        labels = WeisfeilerLehman().fit_transform(adjacency)
        self.assertTrue((labels == np.array([1, 0, 0, 0, 0])).all())

    def test_iso(self):
        adjacency = house()
        n = adjacency.indptr.shape[0] - 1
        reorder = list(range(n))
        np.random.shuffle(reorder)
        adjacency2 = adjacency[reorder][:, reorder]
        l1 = WeisfeilerLehman().fit_transform(adjacency)
        l2 = WeisfeilerLehman().fit_transform(adjacency2)
        l1.sort()
        l2.sort()
        self.assertTrue((l1 == l2).all())

    def test_early_stop(self):
        adjacency = house()
        wl = WeisfeilerLehman(max_iter=1)
        labels = wl.fit_transform(adjacency)
        self.assertTrue((labels == np.array([0, 1, 0, 0, 1])).all())

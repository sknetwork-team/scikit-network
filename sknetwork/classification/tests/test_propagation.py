#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for label propagation"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.classification import Propagation
from sknetwork.data.test_graphs import *


class TestLabelPropagation(unittest.TestCase):

    def test_weighted_neighbor_votes(self):
        adjacency = sparse.csr_matrix([[0, 0, 0, 2, 8],
                                       [0, 0, 0, 7, 1],
                                       [0, 0, 0, 4, 6],
                                       [2, 7, 4, 0, 0],
                                       [8, 1, 6, 0, 0]])
        for weighted, expected in [(True, [1, 0, 1, 0, 1]),
                                   (False, [0, 0, 0, 0, 1])]:
            with self.subTest(weighted=weighted):
                model = Propagation(node_order='index', weighted=weighted)
                labels = model.fit_predict(adjacency, {3: 0, 4: 1})
                np.testing.assert_array_equal(labels, expected)

    def test_degree_order_fixed_labels(self):
        adjacency = sparse.csr_matrix([[0, 0, 1, 0, 0],
                                       [0, 0, 1, 0, 0],
                                       [1, 1, 0, 1, 1],
                                       [0, 0, 1, 0, 0],
                                       [0, 0, 1, 0, 0]])
        for order in ['increasing', 'decreasing']:
            with self.subTest(order=order):
                model = Propagation(node_order=order)
                labels = model.fit_predict(adjacency, {1: 0, 2: 1})
                np.testing.assert_array_equal(labels, [1, 0, 1, 1, 1])

    def test_degree_order_without_fixed_labels(self):
        adjacency = sparse.csr_matrix([[0, 0, 1, 0, 0],
                                       [0, 0, 1, 0, 0],
                                       [1, 1, 0, 1, 1],
                                       [0, 0, 1, 0, 0],
                                       [0, 0, 1, 0, 0]])
        for order, label in [('increasing', 2), ('decreasing', 0)]:
            with self.subTest(order=order):
                labels = Propagation(node_order=order).fit_predict(adjacency)
                np.testing.assert_array_equal(labels, np.full(5, label))

    def test_algo(self):
        for adjacency in [test_graph(), test_digraph(), test_bigraph()]:
            n = adjacency.shape[0]
            labels = {0: 0, 1: 1}
            propagation = Propagation(n_iter=3, weighted=False)
            labels_pred = propagation.fit_predict(adjacency, labels)
            self.assertEqual(labels_pred.shape, (n,))

            for order in ['random', 'decreasing', 'increasing']:
                propagation = Propagation(node_order=order)
                labels_pred = propagation.fit_predict(adjacency, labels)
                self.assertEqual(labels_pred.shape, (n,))

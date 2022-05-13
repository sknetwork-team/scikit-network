#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for optimizer"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.gnn_classifier import GNNClassifier
from sknetwork.gnn.optimizer import optimizer_factory


class TestOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        self.adjacency = test_graph()
        self.features = self.adjacency
        self.labels = np.array([0] * 5 + [1] * 5)

    def test_optimizer_factory(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2)
        with self.assertRaises(ValueError):
            optimizer_factory(gnn, 'toto')

    def test_optimizer_adam(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2, opt='Adam')
        _ = gnn.fit_transform(self.adjacency, self.features, self.labels, max_iter=1)
        conv1_W, conv2_W = gnn.conv1.W.copy(), gnn.conv2.W.copy()
        conv1_b, conv2_b = gnn.conv1.bias.copy(), gnn.conv2.bias.copy()
        gnn.opt.step()
        # Test weight matrix
        self.assertTrue(gnn.conv1.W.shape == conv1_W.shape)
        self.assertTrue(gnn.conv2.W.shape == conv2_W.shape)
        self.assertTrue((gnn.conv1.W != conv1_W).any())
        self.assertTrue((gnn.conv2.W != conv2_W).any())
        # Test bias vector
        self.assertTrue(gnn.conv1.bias.shape == conv1_b.shape)
        self.assertTrue(gnn.conv2.bias.shape == conv2_b.shape)
        self.assertTrue((gnn.conv1.bias != conv1_b).any())
        self.assertTrue((gnn.conv2.bias != conv2_b).any())

    def test_optimizer_gd(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2, opt='none')
        _ = gnn.fit_transform(self.adjacency, self.features, self.labels, max_iter=1)
        conv1_W, conv2_W = gnn.conv1.W.copy(), gnn.conv2.W.copy()
        conv1_b, conv2_b = gnn.conv1.bias.copy(), gnn.conv2.bias.copy()
        gnn.opt.step()
        # Test weight matrix
        self.assertTrue(gnn.conv1.W.shape == conv1_W.shape)
        self.assertTrue(gnn.conv2.W.shape == conv2_W.shape)
        self.assertTrue((gnn.conv1.W != conv1_W).any())
        self.assertTrue((gnn.conv2.W != conv2_W).any())
        # Test bias vector
        self.assertTrue(gnn.conv1.bias.shape == conv1_b.shape)
        self.assertTrue(gnn.conv2.bias.shape == conv2_b.shape)
        self.assertTrue((gnn.conv1.bias != conv1_b).any())
        self.assertTrue((gnn.conv2.bias != conv2_b).any())
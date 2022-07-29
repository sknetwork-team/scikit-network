#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for optimizer"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.gnn_classifier import GNNClassifier
from sknetwork.gnn.optimizer import get_optimizer


class TestOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        self.adjacency = test_graph()
        self.features = self.adjacency
        self.labels = np.array([0] * 5 + [1] * 5)

    def test_get_optimizer(self):
        gnn = GNNClassifier(2)
        with self.assertRaises(ValueError):
            get_optimizer(gnn, 'toto')

    def test_optimizer_adam(self):
        gnn = GNNClassifier([4, 2], 'GCNConv',  ['Relu', 'Softmax'], optimizer='Adam')
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, n_epochs=1, val_size=0.2)
        conv1_weight, conv2_weight = gnn.conv1.weight.copy(), gnn.conv2.weight.copy()
        conv1_b, conv2_b = gnn.conv1.bias.copy(), gnn.conv2.bias.copy()
        gnn.opt.step()
        # Test weight matrix
        self.assertTrue(gnn.conv1.weight.shape == conv1_weight.shape)
        self.assertTrue(gnn.conv2.weight.shape == conv2_weight.shape)
        self.assertTrue((gnn.conv1.weight != conv1_weight).any())
        self.assertTrue((gnn.conv2.weight != conv2_weight).any())
        # Test bias vector
        self.assertTrue(gnn.conv1.bias.shape == conv1_b.shape)
        self.assertTrue(gnn.conv2.bias.shape == conv2_b.shape)
        self.assertTrue((gnn.conv1.bias != conv1_b).any())
        self.assertTrue((gnn.conv2.bias != conv2_b).any())

    def test_optimizer_gd(self):
        gnn = GNNClassifier([4, 2], ['GCNConv', 'GCNConv'], ['Relu', 'Softmax'], optimizer='None')
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, n_epochs=1, val_size=0.2)
        conv1_weight, conv2_weight = gnn.conv1.weight.copy(), gnn.conv2.weight.copy()
        conv1_b, conv2_b = gnn.conv1.bias.copy(), gnn.conv2.bias.copy()
        gnn.opt.step()
        # Test weight matrix
        self.assertTrue(gnn.conv1.weight.shape == conv1_weight.shape)
        self.assertTrue(gnn.conv2.weight.shape == conv2_weight.shape)
        self.assertTrue((gnn.conv1.weight != conv1_weight).any())
        self.assertTrue((gnn.conv2.weight != conv2_weight).any())
        # Test bias vector
        self.assertTrue(gnn.conv1.bias.shape == conv1_b.shape)
        self.assertTrue(gnn.conv2.bias.shape == conv2_b.shape)
        self.assertTrue((gnn.conv1.bias != conv1_b).any())
        self.assertTrue((gnn.conv2.bias != conv2_b).any())

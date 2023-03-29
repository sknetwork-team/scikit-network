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
        self.labels = np.array(4 * [0, 1] + 2 * [-1])

    def test_get_optimizer(self):
        with self.assertRaises(ValueError):
            get_optimizer('foo')
        with self.assertRaises(TypeError):
            get_optimizer(GNNClassifier())

    def test_optimizer(self):
        for optimizer in ['Adam', 'GD']:
            gnn = GNNClassifier([4, 2], 'Conv',  ['Relu', 'Softmax'], optimizer=optimizer)
            _ = gnn.fit_predict(self.adjacency, self.features, self.labels, n_epochs=1)
            conv0_weight, conv1_weight = gnn.layers[0].weight.copy(), gnn.layers[1].weight.copy()
            conv0_b, conv1_b = gnn.layers[0].bias.copy(), gnn.layers[1].bias.copy()
            gnn.optimizer.step(gnn)
            # Test weight matrix
            self.assertTrue(gnn.layers[0].weight.shape == conv0_weight.shape)
            self.assertTrue(gnn.layers[1].weight.shape == conv1_weight.shape)
            self.assertTrue((gnn.layers[0].weight != conv0_weight).any())
            self.assertTrue((gnn.layers[1].weight != conv1_weight).any())
            # Test bias vector
            self.assertTrue(gnn.layers[0].bias.shape == conv0_b.shape)
            self.assertTrue(gnn.layers[1].bias.shape == conv1_b.shape)
            self.assertTrue((gnn.layers[0].bias != conv0_b).any())
            self.assertTrue((gnn.layers[1].bias != conv1_b).any())

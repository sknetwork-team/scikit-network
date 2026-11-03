#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for gradient descent with optional layer biases."""
import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph as make_test_graph
from sknetwork.gnn.gnn_classifier import GNNClassifier


class TestGradientDescentBias(unittest.TestCase):

    def setUp(self):
        random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, random_state)

    def test_training_with_optional_bias(self):
        """Training updates weights and enabled biases while disabled biases stay absent."""
        adjacency = make_test_graph()
        labels = np.array(4 * [0, 1] + 2 * [-1])
        for dims, use_bias in [(2, False), ([4, 2], False), ([4, 2], [True, False]),
                               ([4, 2], [False, True]), ([4, 2], True)]:
            with self.subTest(dims=dims, use_bias=use_bias):
                gnn = GNNClassifier(dims=dims, optimizer='GD', use_bias=use_bias, early_stopping=False)
                gnn.fit(adjacency, adjacency, labels, n_epochs=0, random_state=42)
                initial_weights = [layer.weight.copy() for layer in gnn.layers]
                gnn.fit(adjacency, adjacency, labels, n_epochs=3)
                self.assertEqual(len(gnn.history_['loss']), 3)
                self.assertTrue(np.isfinite(gnn.output_).all())
                for layer, initial_weight in zip(gnn.layers, initial_weights):
                    self.assertEqual(layer.weight.shape, initial_weight.shape)
                    self.assertTrue(np.isfinite(layer.weight).all())
                    self.assertFalse(np.array_equal(layer.weight, initial_weight))
                    if layer.use_bias:
                        self.assertTrue(np.isfinite(layer.bias).all())
                        self.assertTrue(np.any(layer.bias != 0))
                    else:
                        self.assertIsNone(layer.bias)

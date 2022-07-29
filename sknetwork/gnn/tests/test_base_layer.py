#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base layer gnn"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.base_layer import BaseLayer


class TestBaseLayer(unittest.TestCase):

    def setUp(self) -> None:
        """Test graph for tests."""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]
        self.features = self.adjacency
        self.labels = np.array([0]*5 + [1]*5)
        self.base_layer = BaseLayer(len(self.labels))

    def test_base_layer_init(self):
        with self.assertRaises(NotImplementedError):
            self.base_layer.forward(self.adjacency, self.features)

    def test_base_layer_initialize_weights(self):
        self.base_layer._initialize_weights(10)
        self.assertTrue(self.base_layer.weight.shape == (10, len(self.labels)))
        self.assertTrue(all(self.base_layer.bias[0] == np.zeros((len(self.labels), 1)).T[0]))
        self.assertTrue(self.base_layer.weights_initialized)

    def test_base_layer_repr(self):
        base_layers_str = "  BaseLayer(out_channels: 10, activation: relu, use_bias: True, self_loops: True)"
        self.assertTrue(self.base_layer.__repr__() == base_layers_str)
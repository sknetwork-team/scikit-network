#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for layers"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.layers import GCNConv


class TestLayer(unittest.TestCase):

    def setUp(self) -> None:
        """Test graph for tests."""
        self.adjacency = test_graph()
        self.features = self.adjacency
        self.labels = np.array([0] * 5 + [1] * 5)

    def test_graph_conv_shapes(self):
        conv1 = GCNConv(4)
        conv1._initialize_weights(self.features.shape[1])
        conv2 = GCNConv(2)
        conv2._initialize_weights(4)

        self.assertTrue(conv1.weight.shape == (self.features.shape[1], 4))
        self.assertTrue(conv1.bias.shape == (1, 4))
        self.assertTrue(conv1.weights_initialized)
        self.assertTrue(conv2.weight.shape == (4, 2))
        self.assertTrue(conv2.bias.shape == (1, 2))
        self.assertTrue(conv2.weights_initialized)

        h = conv1.forward(self.adjacency, self.features)
        self.assertTrue(h.shape == (self.adjacency.shape[0], 4))
        emb = conv2.forward(self.adjacency, h)
        self.assertTrue(emb.shape == (self.adjacency.shape[0], 2))

    def test_graph_conv_bias_use(self):
        conv1 = GCNConv(4, use_bias=False)
        conv2 = GCNConv(2, use_bias=False)
        h = conv1.forward(self.adjacency, self.features)
        emb = conv2.forward(self.adjacency, h)
        self.assertTrue(emb.shape == (self.adjacency.shape[0], 2))

    def test_graph_conv_self_loops(self):
        conv1 = GCNConv(4, self_loops=False)
        conv2 = GCNConv(2, self_loops=False)
        h = conv1.forward(self.adjacency, self.features)
        emb = conv2.forward(self.adjacency, h)
        self.assertTrue(emb.shape == (self.adjacency.shape[0], 2))

    def test_graph_conv_norm(self):
        conv1 = GCNConv(4, normalization='left')
        conv2 = GCNConv(2, normalization='right')
        h = conv1.forward(self.adjacency, self.features)
        emb = conv2.forward(self.adjacency, h)
        self.assertTrue(emb.shape == (self.adjacency.shape[0], 2))

    def test_graph_conv_activation(self):
        activations = ['Relu', 'Sigmoid']
        for a in activations:
            conv1 = GCNConv(4, activation=a)
            conv2 = GCNConv(2)
            h = conv1.forward(self.adjacency, self.features)
            emb = conv2.forward(self.adjacency, h)
            self.assertTrue(emb.shape == (self.adjacency.shape[0], 2))

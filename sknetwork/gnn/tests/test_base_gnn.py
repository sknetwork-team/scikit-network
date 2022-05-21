#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base gnn"""

import unittest
from abc import ABC

import numpy as np
from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.gnn_classifier import GNNClassifier
from sknetwork.gnn.base_gnn import BaseGNNClassifier
from sknetwork.gnn.layers import GCNConv


class CustomGNNClassifier(GNNClassifier):
    def __init__(self, in_channels: int, h_channels: int, num_classes: int, opt: str):
        super(CustomGNNClassifier, self).__init__(in_channels, h_channels, num_classes, opt)
        self.conv1 = GCNConv(in_channels, h_channels)
        self.conv2 = GCNConv(h_channels, h_channels)
        self.conv3 = GCNConv(h_channels, num_classes, activation='softmax')

    def forward(self, adjacency, feat):
        h = self.conv1(adjacency, feat)
        h = self.conv2(adjacency, h)
        h = self.conv3(adjacency, h)
        return h


class TestBaseGNN(unittest.TestCase):

    def setUp(self) -> None:
        """Test graph for tests."""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]
        self.features = self.adjacency
        self.labels = np.array([0]*5 + [1]*5)

    def test_base_gnn_fit(self):
        gnn = BaseGNNClassifier()
        with self.assertRaises(NotImplementedError):
            gnn.fit(self.adjacency, self.features, self.labels, test_size=0.2)

    def test_base_gnn_custom(self):
        gnn = CustomGNNClassifier(self.features.shape[1], 4, 2, opt='Adam')
        self.assertTrue(isinstance(gnn, CustomGNNClassifier))
        self.assertTrue(gnn.conv3.activation == 'softmax')
        y_pred = gnn.fit_transform(self.adjacency, self.features, self.labels, max_iter=1, test_size=0.2)
        embedding = gnn.conv3.emb
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_base_gnn_repr(self):
        gnn = GNNClassifier(self.features.shape[1], 4, 2, opt='Adam')
        layers_str = "    GCNConv(in_channels: 10, out_channels: 4, use_bias: True, norm: both, self_loops: True, " \
                     "activation: sigmoid)\n" \
                     "    GCNConv(in_channels: 4, out_channels: 2, use_bias: True, norm: both, self_loops: True, " \
                     "activation: softmax)"
        gnn_str = f"GNNClassifier(\n{layers_str}\n)"
        self.assertTrue(gnn.__repr__() == gnn_str)

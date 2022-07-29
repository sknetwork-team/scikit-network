#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base gnn"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.base import BaseGNNClassifier
from sknetwork.gnn.gnn_classifier import GNNClassifier
from sknetwork.gnn.layers import GCNConv


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

    def test_base_gnn_fit_transform(self):
        gnn = GNNClassifier(dims=2, layers='GCNConv', activations='Relu', optimizer='None', verbose=False)
        embedding = gnn.fit_transform(self.adjacency, self.features, labels=self.labels, n_epochs=1, val_size=0.2)
        self.assertTrue(len(embedding) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_base_gnn_custom(self):
        gnn = GNNClassifier(dims=[20, 8, 2], layers='GCNConv',
                            activations=['Relu', 'Sigmoid', 'Softmax'], optimizer='Adam', verbose=False)
        self.assertTrue(isinstance(gnn, GNNClassifier))
        self.assertTrue(gnn.conv3.activation == 'softmax')
        y_pred = gnn.fit_predict(self.adjacency, self.features, labels=self.labels, n_epochs=1, val_size=0.2)
        self.assertTrue(len(y_pred) == self.n)

    def test_base_check_fitted(self):
        gnn = BaseGNNClassifier()
        with self.assertRaises(ValueError):
            gnn._check_fitted()
        gnn = GNNClassifier(dims=2, layers='GCNConv', activations='Relu', optimizer='None', verbose=False)
        gnn.fit_transform(self.adjacency, self.features, labels=self.labels, n_epochs=1, val_size=0.2)
        fit_gnn = gnn._check_fitted()
        self.assertTrue(isinstance(fit_gnn, GNNClassifier))
        self.assertTrue(fit_gnn.embedding_ is not None)

    def test_base_gnn_repr(self):
        gnn = GNNClassifier(dims=[8, 2], layers=['GCNConv', 'GCNConv'],
                            activations=['Relu', 'Softmax'], optimizer='Adam')
        layers_str = "    GCNConv(out_channels: 8, activation: relu, use_bias: True, self_loops: True)\n" \
                     "    GCNConv(out_channels: 2, activation: softmax, use_bias: True, self_loops: True)"
        gnn_str = f"GNNClassifier(\n{layers_str}\n)"
        self.assertTrue(gnn.__repr__() == gnn_str)

    def test_base_gnn_init_layer(self):
        gnn = BaseGNNClassifier()
        with self.assertRaises(ValueError):
            gnn._init_layer('toto')
        gcnconv = gnn._init_layer('GCNConv', 4, 'relu')
        self.assertTrue(isinstance(gcnconv, GCNConv))

    def test_base_gnn_predict(self):
        gnn = BaseGNNClassifier()
        with self.assertRaises(NotImplementedError):
            gnn.predict()

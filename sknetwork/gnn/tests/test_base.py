#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for base gnn"""

import unittest

import numpy as np

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.base import BaseGNN
from sknetwork.gnn.gnn_classifier import GNNClassifier
from sknetwork.gnn.layer import Convolution
from sknetwork.gnn.optimizer import ADAM


class TestBaseGNN(unittest.TestCase):

    def setUp(self) -> None:
        """Test graph for tests."""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]
        self.features = self.adjacency
        self.labels = np.array(4 * [0, 1] + 2 * [-1])

    def test_base_gnn_fit(self):
        gnn = BaseGNN()
        with self.assertRaises(NotImplementedError):
            gnn.fit(self.adjacency, self.features, self.labels, test_size=0.2)

    def test_gnn_fit_transform(self):
        gnn = GNNClassifier(dims=2, layer_types='Conv', activations='Relu', optimizer='GD', verbose=False)
        embedding = gnn.fit_transform(self.adjacency, self.features, labels=self.labels, n_epochs=1)
        self.assertTrue(len(embedding) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_custom_optimizer(self):
        gnn = GNNClassifier(dims=2, layer_types='Conv', activations='Relu', optimizer=ADAM(beta1=0.5), verbose=False)
        embedding = gnn.fit_transform(self.adjacency, self.features, labels=self.labels, n_epochs=1)
        self.assertTrue(len(embedding) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_custom_layers(self):
        gnn = GNNClassifier(layers=[Convolution('Conv', 2, loss='CrossEntropy')], optimizer=ADAM(beta1=0.5))
        embedding = gnn.fit_transform(self.adjacency, self.features, labels=self.labels, n_epochs=1)
        self.assertTrue(len(embedding) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

        gnn = GNNClassifier(layers=[Convolution('SAGEConv', 2, sample_size=5, loss='CrossEntropy')],
                            optimizer=ADAM(beta1=0.5),)
        embedding = gnn.fit_transform(self.adjacency, self.features, labels=self.labels, n_epochs=1)
        self.assertTrue(len(embedding) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_custom(self):
        gnn = GNNClassifier(dims=[20, 8, 2], layer_types='conv',
                            activations=['Relu', 'Sigmoid', 'Softmax'], optimizer='Adam')
        self.assertTrue(isinstance(gnn, GNNClassifier))
        self.assertTrue(gnn.layers[-1].activation.name == 'Cross entropy')
        y_pred = gnn.fit_predict(self.adjacency, self.features, labels=self.labels, n_epochs=1)
        self.assertTrue(len(y_pred) == self.n)

    def test_check_fitted(self):
        gnn = BaseGNN()
        with self.assertRaises(ValueError):
            gnn._check_fitted()
        gnn = GNNClassifier(dims=2, layer_types='conv', activations='Relu', optimizer='GD')
        gnn.fit_transform(self.adjacency, self.features, labels=self.labels, n_epochs=1)
        fit_gnn = gnn._check_fitted()
        self.assertTrue(isinstance(fit_gnn, GNNClassifier))
        self.assertTrue(fit_gnn.embedding_ is not None)

    def test_base_gnn_repr(self):
        gnn = GNNClassifier(dims=[8, 2], layer_types='conv', activations=['Relu', 'Softmax'], optimizer='Adam')
        self.assertTrue(gnn.__repr__().startswith("GNNClassifier"))


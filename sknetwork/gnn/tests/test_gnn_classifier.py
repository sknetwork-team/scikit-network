#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for gnn classifier"""

import unittest

import numpy as np
from scipy import sparse

from sknetwork.data.test_graphs import test_graph
from sknetwork.gnn.gnn_classifier import GNNClassifier


class TestGNNClassifier(unittest.TestCase):

    def setUp(self) -> None:
        """Test graph for tests."""
        self.adjacency = test_graph()
        self.n = self.adjacency.shape[0]
        self.features = self.adjacency
        self.labels = np.array(4 * [0, 1] + 2 * [-1])

    def test_gnn_classifier_sparse_feat(self):
        gnn = GNNClassifier([3, 2], 'Conv', 'Softmax')
        self.assertTrue(gnn.layers[0].activation.name == 'Softmax')
        self.assertTrue(gnn.layers[1].activation.name == 'Cross entropy')
        labels_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
        embedding = gnn.embedding_
        self.assertTrue(len(labels_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_dense_feat(self):
        # features not in nparray
        features = self.adjacency.todense()
        gnn = GNNClassifier(2)
        with self.assertRaises(TypeError):
            gnn.fit_predict(self.adjacency, features, self.labels)

        # features in numpy array
        features = np.array(self.adjacency.todense())
        gnn = GNNClassifier(2, 'Conv')
        y_pred = gnn.fit_predict(self.adjacency, features, self.labels, validation=0.2)
        embedding = gnn.embedding_
        self.assertTrue(len(y_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_no_bias(self):
        gnn = GNNClassifier([3, 2], 'Conv', 'Softmax', use_bias=[True, False])
        labels_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
        embedding = gnn.embedding_
        self.assertTrue(len(labels_pred) == self.n)
        self.assertTrue(embedding.shape == (self.n, 2))
        self.assertTrue(gnn.layers[1].bias is None)

    def test_gnn_classifier_optimizer(self):
        optimizers = ['GD', 'Adam']
        for optimizer in optimizers:
            gnn = GNNClassifier(2, 'Conv', optimizer=optimizer)
            y_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
            embedding = gnn.embedding_
            self.assertTrue(len(y_pred) == self.n)
            self.assertTrue(embedding.shape == (self.n, 2))

    def test_gnn_classifier_binary(self):
        gnn = GNNClassifier([5, 1], 'Conv', 'Softmax')
        self.assertTrue(gnn.layers[1].activation.name == 'Binary cross entropy')
        labels_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
        self.assertTrue(len(labels_pred) == self.n)

    def test_gnn_classifier_norm(self):
        n_labels = len(set(self.labels))
        gnn = GNNClassifier([5, n_labels], 'Conv', normalizations=['left', 'both'])
        labels_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
        self.assertTrue(len(labels_pred) == self.n)

    def test_gnn_classifier_1label(self):
        gnn = GNNClassifier(1, 'Conv', 'Relu')
        labels_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
        self.assertTrue(len(labels_pred) == self.n)

    def test_gnn_classifier_validation(self):
        gnn = GNNClassifier(2, 'Conv', 'Softmax', early_stopping=False)
        labels_pred = gnn.fit_predict(self.adjacency, self.features, self.labels, validation=0.5, random_state=42)
        self.assertTrue(len(labels_pred) == self.n)

    def test_gnn_classifier_dim_output(self):
        gnn = GNNClassifier(2)
        labels = np.arange(len(self.labels))
        with self.assertRaises(ValueError):
            gnn.fit(self.adjacency, self.features, labels)

    def test_gnn_classifier_verbose(self):
        gnn = GNNClassifier(2, verbose=True)
        self.assertTrue(isinstance(gnn, GNNClassifier))

    def test_gnn_classifier_early_stopping(self):
        gnn = GNNClassifier(2, patience=2)
        labels = {0: 0, 1: 1}
        _ = gnn.fit_predict(self.adjacency, self.features, labels, n_epochs=100, validation=0.5,
                            random_state=42)
        self.assertTrue(len(gnn.history_['val_accuracy']) < 100)

        gnn = GNNClassifier(2, early_stopping=False)
        _ = gnn.fit_predict(self.adjacency, self.features, labels, n_epochs=100, validation=0.5,
                            random_state=42)
        self.assertTrue(len(gnn.history_['val_accuracy']) == 100)

    def test_gnn_classifier_reinit(self):
        gnn = GNNClassifier([4, 2])
        gnn.fit(self.adjacency, self.features, self.labels)
        gnn.fit(self.adjacency, self.features, self.labels, n_epochs=1, reinit=True)
        self.assertTrue(gnn.embedding_.shape == (self.n, 2))

    def test_gnn_classifier_sageconv(self):
        gnn = GNNClassifier([4, 2], ['SAGEConv', 'SAGEConv'], sample_sizes=[5, 3])
        _ = gnn.fit_predict(self.adjacency, self.features, self.labels, n_epochs=100)
        self.assertTrue(gnn.layers[0].sample_size == 5 and gnn.layers[0].normalization == 'left')
        self.assertTrue(gnn.layers[1].sample_size == 3 and gnn.layers[1].normalization == 'left')

    def test_gnn_classifier_predict(self):
        gnn = GNNClassifier([4, 2])
        labels_pred = gnn.fit_predict(self.adjacency, self.features, self.labels)
        labels_pred_ = gnn.predict()
        self.assertTrue(all(labels_pred == gnn.labels_))
        self.assertTrue(all(labels_pred == labels_pred_))

    def test_gnn_classifier_predict_proba(self):
        gnn = GNNClassifier([4, 2])
        probs = gnn.fit_predict_proba(self.adjacency, self.features, self.labels)
        self.assertTrue(probs.shape[1] == 2)

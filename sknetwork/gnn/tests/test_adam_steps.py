#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adam bias correction uses optimization time, independently of layer count."""

import unittest

import numpy as np

from sknetwork.data import karate_club
from sknetwork.gnn import GNNClassifier
from sknetwork.gnn.optimizer import ADAM


class TestAdamSteps(unittest.TestCase):

    def setUp(self):
        random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, random_state)

    def test_first_training_step(self):
        """Every layer takes Adam's first update during the first training epoch."""
        graph = karate_club(metadata=True)
        for dims in (2, [4, 2], [4, 3, 2]):
            with self.subTest(dims=dims):
                frozen = GNNClassifier(dims, learning_rate=0, early_stopping=False)
                trained = GNNClassifier(dims, learning_rate=0.1, early_stopping=False)
                for model in (frozen, trained):
                    model.fit(graph.adjacency, graph.adjacency, graph.labels,
                              n_epochs=1, random_state=42)
                for idx, (before, after) in enumerate(zip(frozen.layers, trained.layers)):
                    for name in ('weight', 'bias'):
                        gradients = getattr(frozen, 'derivative_' + name)[idx]
                        expected = getattr(before, name) - 0.1 * gradients / (np.abs(gradients) + trained.optimizer.eps)
                        np.testing.assert_allclose(getattr(after, name), expected, rtol=1e-12, atol=1e-12)

    def test_updates_match_weighted_gradient_history(self):
        """Changing gradients use the same bias-correction time in every layer."""
        weight_gradients = np.array([[[0.4, -0.2], [0, 0.3]],
                                     [[-0.1, 0.5], [0.2, -0.4]],
                                     [[0.3, 0.1], [-0.2, 0]]])
        bias_gradients = np.array([[[0.5, -0.25]], [[-0.1, 0.3]], [[0.2, 0.4]]])
        for betas in ((0.9, 0.999), (0.6, 0.8)):
            for n_layers in (1, 2, 3):
                with self.subTest(betas=betas, n_layers=n_layers):
                    optimizer = ADAM(learning_rate=0.03, beta1=betas[0], beta2=betas[1], eps=1e-7)
                    model = GNNClassifier([2] * n_layers, optimizer=optimizer)
                    expected_weights = [np.full((2, 2), 1 + idx / 10) for idx in range(n_layers)]
                    expected_biases = [np.array([[0.2, -0.1]]) + idx for idx in range(n_layers)]
                    for idx, layer in enumerate(model.layers):
                        layer.weight = expected_weights[idx].copy()
                        layer.bias = expected_biases[idx].copy()
                    for step in range(1, 4):
                        model.derivative_weight = [weight_gradients[step - 1] * (idx + 1) for idx in range(n_layers)]
                        model.derivative_bias = [bias_gradients[step - 1] * (idx + 1) for idx in range(n_layers)]
                        optimizer.step(model)
                        powers = np.arange(step - 1, -1, -1)
                        first_weights = (1 - betas[0]) * betas[0] ** powers / (1 - betas[0] ** step)
                        second_weights = (1 - betas[1]) * betas[1] ** powers / (1 - betas[1] ** step)
                        for idx, layer in enumerate(model.layers):
                            for name, history, expected in (('weight', weight_gradients, expected_weights),
                                                            ('bias', bias_gradients, expected_biases)):
                                gradients = history[:step] * (idx + 1)
                                first = np.einsum('i,ijk->jk', first_weights, gradients)
                                second = np.einsum('i,ijk->jk', second_weights, gradients ** 2)
                                expected[idx] -= 0.03 * first / (np.sqrt(second) + optimizer.eps)
                                np.testing.assert_allclose(getattr(layer, name), expected[idx], rtol=1e-12, atol=1e-12)

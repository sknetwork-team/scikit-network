#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for activation"""

import unittest

import numpy as np

from sknetwork.gnn.activation import *


class TestActivation(unittest.TestCase):

    def test_activation_relu(self):
        a = np.linspace(-2, 2, 5)
        self.assertTrue((relu(a) == [0., 0., 0., 1., 2.]).all())

    def test_activation_relu_prime(self):
        a = np.array([[-2, 1, 4], [0, 3, 4], [-3, -2, -1]])
        self.assertTrue((relu_prime(a) == np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])).all())

    def test_activation_sigmoid(self):
        a = np.array([-np.inf, -1.5, 0, 1.5, np.inf])
        self.assertTrue(np.allclose(sigmoid(a), np.array([0., 0.18242552, 0.5, 0.81757448, 1.])))

    def test_activation_sigmoid_prime(self):
        a = np.array([-np.inf, 0, np.inf])
        print(sigmoid_prime(a))
        self.assertTrue(np.allclose(sigmoid_prime(a), np.array([0., 0.25, 0.])))

    def test_activation_softmax(self):
        a = np.array([[-1, 0, 3, 5], [-1, 0, 3, 5]])
        self.assertTrue(np.allclose(softmax(a),
                                    np.array([[0.0021657, 0.00588697, 0.11824302, 0.87370431],
                                              [0.0021657, 0.00588697, 0.11824302, 0.87370431]])))
        self.assertAlmostEqual((softmax(a).sum(axis=1) - np.array([1, 1])).sum(), 0)

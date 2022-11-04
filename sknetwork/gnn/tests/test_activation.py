#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for activation"""

import unittest

from sknetwork.gnn.activation import *


class TestActivation(unittest.TestCase):

    def test_get_activation(self):
        self.assertTrue(isinstance(get_activation('Identity'), BaseActivation))
        self.assertTrue(isinstance(get_activation('Relu'), ReLu))
        self.assertTrue(isinstance(get_activation('Sigmoid'), Sigmoid))
        self.assertTrue(isinstance(get_activation('Softmax'), Softmax))
        with self.assertRaises(ValueError):
            get_activation('foo')

        base_act = BaseActivation()
        self.assertTrue(base_act == get_activation(base_act))
        with self.assertRaises(TypeError):
            get_activation(0)

    def test_activation_identity(self):
        activation = get_activation('Identity')
        signal = np.arange(5)
        self.assertTrue((activation.output(signal) == signal).all())
        direction = np.arange(5)
        self.assertTrue((activation.gradient(signal, direction) == direction).all())

    def test_activation_relu(self):
        activation = get_activation('ReLu')
        signal = np.linspace(-2, 2, 5)
        self.assertTrue((activation.output(signal) == [0., 0., 0., 1., 2.]).all())
        direction = np.arange(5)
        self.assertTrue((activation.gradient(signal, direction) == direction * (signal > 0)).all())

    def test_activation_sigmoid(self):
        activation = get_activation('Sigmoid')
        signal = np.array([-np.inf, -1.5, 0, 1.5, np.inf])
        self.assertTrue(np.allclose(activation.output(signal), np.array([0., 0.18242552, 0.5, 0.81757448, 1.])))
        signal = np.array([[-1000, 1000, 1000]])
        direction = np.arange(3)
        self.assertTrue(np.allclose(activation.output(signal), np.array([[0., 1., 1.]])))
        self.assertTrue(np.allclose(activation.gradient(signal, direction), np.array([[0., 0., 0.]])))

    def test_activation_softmax(self):
        activation = get_activation('Softmax')
        signal = np.array([[-1, 0, 3, 5]])
        output = activation.output(signal)
        self.assertTrue(np.allclose(output, np.array([[0.0021657, 0.00588697, 0.11824302, 0.87370431]])))
        signal = np.array([[-1000, 1000, 1000]])
        direction = np.arange(3)
        self.assertTrue(np.allclose(activation.output(signal), np.array([[0., 0.5, 0.5]])))
        self.assertTrue(np.allclose(activation.gradient(signal, direction), np.array([[0., -0.25, 0.25]])))

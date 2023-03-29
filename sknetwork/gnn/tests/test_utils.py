#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for gnn utils"""

import unittest

from sknetwork.gnn.utils import *


class TestUtils(unittest.TestCase):

    def test_check_norm(self):
        with self.assertRaises(ValueError):
            check_normalizations('foo')
        with self.assertRaises(ValueError):
            check_normalizations(['foo', 'bar'])

    def test_early_stopping(self):
        self.assertTrue(check_early_stopping(True, np.array([True, False]), 2))
        self.assertFalse(check_early_stopping(True, None, 2))
        self.assertFalse(check_early_stopping(True, np.array([True, False, True]), None))
        self.assertFalse(check_early_stopping(True, np.array([False, False, False]), 5))

    def test_get_layers(self):
        with self.assertRaises(ValueError):
            get_layers([4, 2], 'Conv',  activations=['Relu', 'Sigmoid', 'Relu'], use_bias=True, normalizations='Both',
                       self_embeddings=True, sample_sizes=5, loss=None)
        # Type compatibility
        layers = get_layers([4], 'Conv', activations=['Relu'], use_bias=[True], normalizations=['Both'],
                            self_embeddings=[True], sample_sizes=[5], loss='Cross entropy')
        self.assertTrue(len(np.ravel(layers)) == 1)
        # Broadcasting parameters
        layers = get_layers([4, 2], ['Conv', 'Conv'], activations='Relu', use_bias=True, normalizations='Both',
                            self_embeddings=True, sample_sizes=5, loss='Cross entropy')
        self.assertTrue(len(layers) == 2)

    def test_check_loss(self):
        layer = get_layers([4], 'Conv', activations=['Relu'], use_bias=[True], normalizations=['Both'],
                           self_embeddings=[True], sample_sizes=[5], loss=None)
        with self.assertRaises(ValueError):
            check_loss(layer[0])

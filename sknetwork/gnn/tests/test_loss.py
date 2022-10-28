#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for loss"""

import unittest

from sknetwork.gnn.loss import *


class TestLoss(unittest.TestCase):

    def test_get_loss(self):
        self.assertTrue(isinstance(get_loss('CrossEntropy'), CrossEntropy))
        self.assertTrue(isinstance(get_loss('BinaryCrossEntropy'), BinaryCrossEntropy))
        with self.assertRaises(ValueError):
            get_loss('foo')

        base_loss = BaseLoss()
        self.assertTrue(base_loss == get_loss(base_loss))
        with self.assertRaises(TypeError):
            get_loss(0)

    def test_ce_loss(self):
        cross_entropy = CrossEntropy()
        signal = np.array([[0, 5]])
        labels = np.array([1])
        self.assertAlmostEqual(cross_entropy.loss(signal, labels), 0.00671534848911828)

    def test_bce_loss(self):
        binary_cross_entropy = BinaryCrossEntropy()
        signal = np.array([[0, 5]])
        labels = np.array([1])
        self.assertAlmostEqual(binary_cross_entropy.loss(signal, labels), 0.6998625290490632)

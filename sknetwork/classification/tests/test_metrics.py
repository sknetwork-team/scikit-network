#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for classification metrics"""

import unittest

from sknetwork.classification.metrics import *


class TestMetrics(unittest.TestCase):

    def setUp(self) -> None:
        self.labels_true = np.array([0, 1, 1, 2, 2, -1])
        self.labels_pred1 = np.array([0, -1, 1, 2, 0, 0])
        self.labels_pred2 = np.array([-1, -1, -1, -1, -1, 0])

    def test_accuracy(self):
        self.assertEqual(get_accuracy_score(self.labels_true, self.labels_pred1), 0.75)
        with self.assertRaises(ValueError):
            get_accuracy_score(self.labels_true, self.labels_pred2)

    def test_confusion(self):
        self.assertEqual(get_confusion_matrix(self.labels_true, self.labels_pred1).data.sum(), 4)
        with self.assertRaises(ValueError):
            get_accuracy_score(self.labels_true, self.labels_pred2)

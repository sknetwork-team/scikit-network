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
        confusion = get_confusion_matrix(self.labels_true, self.labels_pred1)
        self.assertEqual(confusion.data.sum(), 4)
        self.assertEqual(confusion.diagonal().sum(), 3)
        with self.assertRaises(ValueError):
            get_accuracy_score(self.labels_true, self.labels_pred2)

    def test_f1_score(self):
        f1_score = get_f1_score(np.array([0, 0, 1]), np.array([0, 1, 1]))
        self.assertAlmostEqual(f1_score, 0.67, 2)
        with self.assertRaises(ValueError):
            get_f1_score(self.labels_true, self.labels_pred1)

    def test_f1_scores(self):
        f1_scores = get_f1_scores(self.labels_true, self.labels_pred1)
        self.assertAlmostEqual(min(f1_scores), 0.67, 2)
        f1_scores, precisions, recalls = get_f1_scores(self.labels_true, self.labels_pred1, True)
        self.assertAlmostEqual(min(f1_scores), 0.67, 2)
        self.assertAlmostEqual(min(precisions), 0.5, 2)
        self.assertAlmostEqual(min(recalls), 0.5, 2)
        with self.assertRaises(ValueError):
            get_f1_scores(self.labels_true, self.labels_pred2)

    def test_average_f1_score(self):
        f1_score = get_average_f1_score(self.labels_true, self.labels_pred1)
        self.assertAlmostEqual(f1_score, 0.78, 2)
        f1_score = get_average_f1_score(self.labels_true, self.labels_pred1, average='micro')
        self.assertEqual(f1_score, 0.75)
        f1_score = get_average_f1_score(self.labels_true, self.labels_pred1, average='weighted')
        self.assertEqual(f1_score, 0.80)
        with self.assertRaises(ValueError):
            get_average_f1_score(self.labels_true, self.labels_pred2, 'toto')

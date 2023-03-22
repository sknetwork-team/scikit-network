#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for PageRankClassifier"""

import unittest

from sknetwork.classification import PageRankClassifier
from sknetwork.data.test_graphs import *


class TestPageRankClassifier(unittest.TestCase):

    def test_solvers(self):
        adjacency = test_graph()
        labels = {0: 0, 1: 1}

        ref = PageRankClassifier(solver='piteration').fit_predict(adjacency, labels)
        for solver in ['lanczos', 'bicgstab']:
            labels_pred = PageRankClassifier(solver=solver).fit_predict(adjacency, labels)
            self.assertTrue((ref == labels_pred).all())

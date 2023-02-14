#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for tfidf.py"""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.utils.tfidf import get_tfidf


class TestTFIDF(unittest.TestCase):

    def test_tfidf(self):
        count = sparse.csr_matrix(np.array([[0, 1, 2], [0, 2, 1], [0, 0, 1]]))
        tfidf = get_tfidf(count)
        self.assertEqual(count.shape, tfidf.shape)
        self.assertEqual(tfidf.nnz, 2)

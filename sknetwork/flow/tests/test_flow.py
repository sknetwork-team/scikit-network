#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for push_relabel.py"""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.flow import get_max_flow, find_excess


class TestFlow(unittest.TestCase):
    def test_push_relabel(self):
        row = [0, 0, 1, 2, 3, 3, 4, 5]
        col = [1, 2, 3, 3, 4, 5, 6, 6]
        data = [2, 3, 3, 2, 1, 3, 2, 2]
        adj = sparse.csr_matrix((data, (row, col)), shape=(7, 7))
        flow = get_max_flow(adj, 0, 6)
        self.assertTrue((flow.data == np.array([1, 2, 1, 2, 1, 2, 1, 2])).all())
        self.assertEqual(3, flow[:, 6].sum())
        for i in range(1, 6):
            self.assertEqual(0, find_excess(flow, i))

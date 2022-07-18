#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""tests for flow.py"""
import unittest

from sknetwork.flow import push_relabel, find_excess
from scipy import sparse

class TestFlow(unittest.TestCase):
    def test_push_relabel_1(self):
        adj = sparse.csr_matrix([[0, 2, 3, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 1, 3, 0], [0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0]])
        flow = push_relabel(adj, 0, 6)
        self.assertEqual(3, flow[:, 6].sum())
        for i in range(1, 6):
            self.assertEqual(0, find_excess(flow, i))

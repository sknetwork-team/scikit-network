#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for data API"""

import tempfile
import unittest
import warnings

from sknetwork.data.load import *
from sknetwork.data.toy_graphs import *
from sknetwork.data import Dataset


class TestDataAPI(unittest.TestCase):

    def test_toy_graphs(self):
        toy_graphs = [karate_club, painters, bow_tie, house, miserables]
        for toy_graph in toy_graphs:
            self.assertEqual(type(toy_graph()), sparse.csr_matrix)
            self.assertEqual(type(toy_graph(metadata=True)), Dataset)

    def test_load(self):
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        clear_data_home(tmp_data_dir)
        try:
            graph = load_netset('stub', tmp_data_dir)
            self.assertEqual(type(graph), Dataset)
        except URLError:  # pragma: no cover
            warnings.warn('Could not reach NetSet. Corresponding test has not been performed.', RuntimeWarning)
            return

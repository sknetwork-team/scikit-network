#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for load.py"""

from urllib.error import URLError
import unittest
import tempfile
import warnings

from sknetwork.data import load_netset, load_konect, clear_data_home


class TestLoader(unittest.TestCase):

    def test_wikilinks(self):
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        clear_data_home(tmp_data_dir)
        try:
            graph = load_netset('stub', tmp_data_dir)
        except URLError:
            warnings.warn('Could not reach Telecom Graphs. Corresponding test has not been performed.', RuntimeWarning)
            return
        n = 2
        self.assertEqual(graph.adjacency.shape, (n, n))
        self.assertEqual(len(graph.names), n)
        clear_data_home(tmp_data_dir)

    def test_invalid_wikilinks(self):
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        clear_data_home(tmp_data_dir)
        try:
            with self.assertRaises(ValueError):
                load_netset('junk', tmp_data_dir)
        except URLError:
            warnings.warn('Could not reach Telecom Graphs. Corresponding test has not been performed.', RuntimeWarning)
            return

    def test_konect(self):
        tmp_data_dir = tempfile.gettempdir() + '/moreno_crime'
        clear_data_home(tmp_data_dir)
        try:
            data = load_konect('moreno_crime', tmp_data_dir)
        except URLError:
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)
        data = load_konect('moreno_crime', tmp_data_dir)
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)
        clear_data_home(tmp_data_dir)

    def test_invalid_konect(self):
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        clear_data_home(tmp_data_dir)
        try:
            with self.assertRaises(ValueError):
                load_konect('junk', tmp_data_dir)
        except URLError:
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return

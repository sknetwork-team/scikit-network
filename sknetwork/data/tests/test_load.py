#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for load.py"""

import tempfile
import unittest
import warnings

import numpy as np

from sknetwork.data.load import load_netset, load_konect, clear_data_home, save, load
from sknetwork.data.toy_graphs import house, star_wars
from sknetwork.data.timeout import TimeOut


class TestLoader(unittest.TestCase):

    def setUp(self):
        self.data_home = tempfile.gettempdir() + '/data'

    def test_netset(self):
        clear_data_home(self.data_home)
        try:
            graph = load_netset('stub', self.data_home)
        except:  # pragma: no cover
            warnings.warn('Could not reach the NetSet collection. Corresponding test has not been performed.',
                          RuntimeWarning)
            return
        n = 2
        self.assertEqual(graph.adjacency.shape, (n, n))
        self.assertEqual(len(graph.names), n)
        clear_data_home(self.data_home)

    def test_invalid_netset(self):
        try:
            with self.assertRaises(ValueError):
                load_netset('junk', self.data_home)
        except:  # pragma: no cover
            warnings.warn('Could not reach the NetSet collection. Corresponding test has not been performed.',
                          RuntimeWarning)
            return
        load_netset()

    def test_konect(self):
        try:
            with TimeOut(2):
                data = load_konect('moreno_crime', self.data_home)
        except (TimeoutError, RuntimeError):  # pragma: no cover
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)

        # load from bundle
        data = load_konect('moreno_crime', self.data_home)
        self.assertEqual(data.biadjacency.shape[0], 829)

        try:
            with TimeOut(2):
                data = load_konect('ego-facebook', self.data_home)
        except (TimeoutError, RuntimeError):  # pragma: no cover
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return
        self.assertEqual(data.adjacency.shape[0], 2888)
        clear_data_home(self.data_home)

    def test_invalid_konect(self):
        try:
            with TimeOut(4):
                with self.assertRaises(ValueError):
                    load_konect('junk', self.data_home)
                with self.assertRaises(ValueError):
                    load_konect('', self.data_home)
        except (TimeoutError, RuntimeError):  # pragma: no cover
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return

    def test_save_load(self):
        data = house()
        save(self.data_home + '/house', data)
        loaded_data = load(self.data_home + '/house')
        self.assertTrue(np.allclose(data.data, loaded_data.adjacency.data))

        data = star_wars()
        save(self.data_home + '/star_wars', data)
        loaded_data = load(self.data_home + '/star_wars')
        self.assertTrue(np.allclose(data.data, loaded_data.biadjacency.data))

        data = star_wars(metadata=True)
        save(self.data_home + '/star_wars', data)
        loaded_data = load(self.data_home + '/star_wars')
        self.assertTrue(np.allclose(data.biadjacency.data, loaded_data.biadjacency.data))
        self.assertEqual(data.names_col[0], loaded_data.names_col[0])



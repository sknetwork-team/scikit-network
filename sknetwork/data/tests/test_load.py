#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for load.py"""

import tempfile
import unittest
import warnings

import numpy as np

from sknetwork.data.load import load_netset, load_konect, clear_data_home, save, load
from sknetwork.data.toy_graphs import house, star_wars
from sknetwork.utils.timeout import TimeOut


class TestLoader(unittest.TestCase):

    def test_netset(self):
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        clear_data_home(tmp_data_dir)
        try:
            graph = load_netset('stub', tmp_data_dir)
        except:  # pragma: no cover
            warnings.warn('Could not reach the NetSet collection. Corresponding test has not been performed.',
                          RuntimeWarning)
            return
        n = 2
        self.assertEqual(graph.adjacency.shape, (n, n))
        self.assertEqual(len(graph.names), n)
        clear_data_home(tmp_data_dir)

    def test_invalid_netset(self):
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        clear_data_home(tmp_data_dir)
        try:
            with self.assertRaises(ValueError):
                load_netset('junk', tmp_data_dir)

        except:  # pragma: no cover
            warnings.warn('Could not reach the NetSet collection. Corresponding test has not been performed.',
                          RuntimeWarning)
            return
        load_netset()

    def test_konect(self):
        tmp_data_dir = tempfile.gettempdir() + '/moreno_crime'
        clear_data_home(tmp_data_dir)
        try:
            with TimeOut(2):
                data = load_konect('moreno_crime', tmp_data_dir)
        except (TimeoutError, RuntimeError):  # pragma: no cover
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)
        data = load_konect('moreno_crime', tmp_data_dir)
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)
        clear_data_home(tmp_data_dir)

        tmp_data_dir = tempfile.gettempdir() + '/ego-facebook'
        try:
            with TimeOut(2):
                data = load_konect('ego-facebook', tmp_data_dir)
        except (TimeoutError, RuntimeError):  # pragma: no cover
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return
        self.assertEqual(data.adjacency.shape[0], 2888)
        data = load_konect('ego-facebook', tmp_data_dir)
        self.assertEqual(data.adjacency.shape[0], 2888)
        clear_data_home(tmp_data_dir)

    def test_invalid_konect(self):
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        clear_data_home(tmp_data_dir)
        try:
            with TimeOut(4):
                with self.assertRaises(ValueError):
                    load_konect('junk', tmp_data_dir)
                with self.assertRaises(ValueError):
                    load_konect('', tmp_data_dir)
        except (TimeoutError, RuntimeError):  # pragma: no cover
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return

    def test_save_load(self):
        data = house()
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        save(tmp_data_dir + '/house', data)
        loaded_data = load(tmp_data_dir + '/house')
        self.assertTrue(np.allclose(data.data, loaded_data.adjacency.data))

        data = star_wars()
        save(tmp_data_dir + '/star_wars', data)
        loaded_data = load(tmp_data_dir + '/star_wars')
        self.assertTrue(np.allclose(data.data, loaded_data.biadjacency.data))

        data = star_wars(metadata=True)
        save(tmp_data_dir + '/star_wars', data)
        loaded_data = load(tmp_data_dir + '/star_wars')
        self.assertTrue(np.allclose(data.biadjacency.data, loaded_data.biadjacency.data))
        self.assertEqual(data.names_col[0], loaded_data.names_col[0])



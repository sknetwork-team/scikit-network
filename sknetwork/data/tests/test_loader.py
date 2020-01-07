#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for loading.py"""

from urllib.error import URLError
import unittest
import tempfile
import warnings

from sknetwork.data import load_wikilinks_dataset, load_konect_dataset, clear_data_home


class TestLoader(unittest.TestCase):

    def test_wikilinks(self):
        tmp_data_dir = tempfile.gettempdir() + '/stub'
        clear_data_home(tmp_data_dir)
        try:
            data = load_wikilinks_dataset('stub', tmp_data_dir)
        except URLError:
            warnings.warn('Could not reach Telecom Graphs. Corresponding test has not been performed.', RuntimeWarning)
            return
        self.assertEqual(data.biadjacency.shape[0], 1)
        self.assertEqual(data.names.shape[0], 2)
        clear_data_home(tmp_data_dir)

    def test_konect(self):
        tmp_data_dir = tempfile.gettempdir() + '/moreno_crime'
        clear_data_home(tmp_data_dir)
        try:
            data = load_konect_dataset('moreno_crime', tmp_data_dir)
        except URLError:
            warnings.warn('Could not reach Konect. Corresponding test has not been performed.', RuntimeWarning)
            return
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)
        data = load_konect_dataset('moreno_crime', tmp_data_dir)
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)
        clear_data_home(tmp_data_dir)

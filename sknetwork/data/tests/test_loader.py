#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for loading.py"""

import unittest
import tempfile

from sknetwork.data import load_wikilinks_dataset, load_konect_dataset, clear_data_home


class TestLoader(unittest.TestCase):

    def test_wiki_vital(self):
        tmp_data_dir = tempfile.gettempdir() + '/vital_wikipedia'
        clear_data_home(tmp_data_dir)
        data = load_wikilinks_dataset('vital_wikipedia', tmp_data_dir)
        self.assertEqual(data.biadjacency.shape[0], 10012)
        self.assertEqual(data.names.shape[0], 10012)
        clear_data_home(tmp_data_dir)

    def test_konect(self):
        tmp_data_dir = tempfile.gettempdir() + '/moreno_crime'
        clear_data_home(tmp_data_dir)
        data = load_konect_dataset('moreno_crime', tmp_data_dir)
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)
        clear_data_home(tmp_data_dir)

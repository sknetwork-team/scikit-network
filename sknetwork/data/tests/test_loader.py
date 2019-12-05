#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for loading.py"""

import unittest
import tempfile

from sknetwork.data import load_vital_wikipedia_links, load_vital_wikipedia_text, load_konect_dataset, clear_data_home


class TestLoader(unittest.TestCase):

    def test_wiki_vital(self):
        tmp_data_dir = tempfile.gettempdir() + '/vital_wikipedia'
        clear_data_home(tmp_data_dir)
        links = load_vital_wikipedia_links(tmp_data_dir)
        text = load_vital_wikipedia_text(tmp_data_dir)
        self.assertTrue((links.names == text.names).all())
        self.assertTrue((links.target == text.target).all())
        clear_data_home(tmp_data_dir)

    def test_konect(self):
        tmp_data_dir = tempfile.gettempdir() + '/moreno_crime'
        clear_data_home(tmp_data_dir)
        data = load_konect_dataset('moreno_crime', tmp_data_dir)
        self.assertEqual(data.biadjacency.shape[0], 829)
        self.assertEqual(data.name.shape[0], 829)
        clear_data_home(tmp_data_dir)

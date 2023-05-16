# -*- coding: utf-8 -*-
"""tests for dataset"""

import unittest

from sknetwork.data.base import Bunch


class TestDataset(unittest.TestCase):

    def test(self):
        dataset = Bunch(name='dataset')
        self.assertEqual(dataset.name, 'dataset')
        self.assertEqual(dataset['name'], 'dataset')

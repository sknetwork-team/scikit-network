#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for verbose.py"""

import unittest

from sknetwork.utils.verbose import VerboseMixin


class TestVerbose(unittest.TestCase):

    def test_prints(self):
        verbose = VerboseMixin(verbose=True)
        verbose.log.print('There are', 4, 'seasons in a year')
        self.assertEqual(str(verbose.log), 'There are 4 seasons in a year\n')

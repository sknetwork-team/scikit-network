#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in December 2019
@author: Quentin Lutz <qlutz@enst.fr>
"""


class Log:
    """Log class for verbosity features"""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.log = ''

    def print_log(self, *args):
        """Fill log with text."""
        if self.verbose:
            print(*args)
        self.log += ' '.join(map(str, args)) + '\n'

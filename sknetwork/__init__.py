#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for scikit-network"""

__author__ = """scikit-network team"""
__email__ = "bonald@enst.fr"
__version__ = '0.7.1'

import numpy as np
try:
    from numba import __version__ as numba_version
    if int(numba_version.split('.')[1]) < 44 and int(numba_version.split('.')[0]) == 0:
        raise ImportWarning('To enable all features using Numba, please update Numba.')
    from numba import njit, prange, types
    from numba.typed import Dict as TypedDict
    is_numba_available = True
except ImportError:
    numba_version = ''

    def njit(*args, **kwargs):
        if len(args) > 0:
            if callable(args[0]):
                return args[0]
        else:
            def __wrapper__(func):
                return func
            return __wrapper__
    prange = range
    is_numba_available = False
    types = np

    class TypedDict(dict):
        @staticmethod
        def empty(**kwargs):
            pass


import sknetwork.clustering
import sknetwork.embedding
import sknetwork.hierarchy
import sknetwork.loader
import sknetwork.ranking
import sknetwork.toy_graphs
import sknetwork.utils

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for scikit-network"""

__author__ = """scikit-network team"""
__email__ = "bonald@enst.fr"
__version__ = '0.8.0'

import numpy as np

import warnings

warnings.filterwarnings("default", category=DeprecationWarning)
try:
    from numba import __version__ as numba_version
    if [int(num) for num in numba_version.split('.')] < [0, 44, 0]:
        raise DeprecationWarning('To enable all features using Numba, please update Numba (currently using {}).'
                                 .format(numba_version))
    from numba import njit, prange, types
    from numba.typed import Dict as TypedDict
    is_numba_available = True
except (ImportError, DeprecationWarning) as error:
    numba_version = ''
    if type(error) is DeprecationWarning:
        warnings.warn(error, DeprecationWarning)

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

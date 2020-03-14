#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for scikit-network"""

__author__ = """scikit-network team"""
__email__ = "bonald@enst.fr"
__version__ = '0.12.1'

import os
import warnings

import numpy as np

warnings.filterwarnings("default", category=DeprecationWarning)
try:
    if os.environ.get('SKNETWORK_DISABLE_NUMBA') == 'true':
        raise ImportError('Will not use Numba.')

    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from numba import __version__ as numba_version

    if [int(num) for num in numba_version.split('.')] < [0, 44, 0]:
        raise DeprecationWarning('To enable all features using Numba, please update Numba (currently using {}).'
                                 .format(numba_version))
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from numba import njit, prange, types
    # noinspection PyPackageRequirements
    from numba.typed import Dict as TypedDict

    is_numba_available = True
except (ImportError, DeprecationWarning) as error:
    numba_version = ''
    if type(error) is DeprecationWarning:
        warnings.warn(error, DeprecationWarning)

    # noinspection PyUnusedLocal
    def njit(*args, **kwargs):
        if len(args) > 0:
            if callable(args[0]):
                return args[0]


    prange = range
    is_numba_available = False
    types = np


    class TypedDict(dict):
        @staticmethod
        def empty(**kwargs):
            pass

import sknetwork.basics
import sknetwork.classification
import sknetwork.clustering
import sknetwork.embedding
import sknetwork.hierarchy
import sknetwork.linalg
import sknetwork.ranking
import sknetwork.soft_classification
import sknetwork.data
import sknetwork.utils

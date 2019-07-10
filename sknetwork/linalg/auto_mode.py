#!/usr/bin/env python3
# coding: utf-8
"""
Created on July 10 2019

Authors:
Nathan De Lara <nathan.delara@telecom-paris.fr>
"""


def auto_solver(shape: tuple) -> str:
    """Recommend a solver for SVD or Eigenvalue decomposition depending of the size of the input matrix.
    Halko's randomized method is returned for big matrices and Lanczos for small ones.

    Parameters
    ----------
    shape: tuple
        Shape of the matrix to decompose.

    Returns
    -------
    solver name: str
        'halko' or ' lanczos'

    """
    if min(shape[0], shape[1]) > 10 ** 3:
        return 'halko'
    else:
        return 'lanczos'

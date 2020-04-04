#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr, 2019
@author: Nathan de Lara <ndelara@enst.fr>
"""
from typing import Optional, Union

import numpy as np

from sknetwork.utils.check import check_seeds


def stack_seeds(n_row: int, n_col: int, seeds_row: Union[np.ndarray, dict],
                seeds_col: Optional[Union[np.ndarray, dict]] = None) -> np.ndarray:
    """Process seeds for rows and columns and stack the results into a single vector."""
    labels_row = check_seeds(seeds_row, n_row).astype(int)
    if seeds_col is None:
        labels_col = -np.ones(n_col, dtype=int)
    else:
        labels_col = check_seeds(seeds_col, n_col).astype(int)
    labels = np.hstack((labels_row, labels_col))

    return labels

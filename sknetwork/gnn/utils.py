#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Optional

import numpy as np

from sknetwork.utils.check import check_is_proba, has_boolean_entries, check_boolean


def check_norm(norm: str):
    """Check if normalization is known.

    Parameters
    ----------
    norm: str
        Normalization kind for adjacency matrix.
    """
    if norm not in ['Both']:
        raise ValueError('Unknown norm parameter.')


def check_existing_masks(train_mask: Optional[np.ndarray] = None, test_mask: Optional[np.ndarray] = None,
                         test_size: Optional[float] = None) -> bool:
    """Check if train/test masks are provided.

    Parameters
    ----------
    train_mask, test_mask: np.ndarray, np.ndarray
        Boolean array indicating whether nodes are in training or test sets.
    test_size: float
        If `train_mask` and `test_mask` are `None`, includes this proportion of the nodes in test set.

    Returns
    -------
        True if train/test masks are provided.
    """

    if train_mask is None or test_mask is None:
        if test_size is None:
            raise ValueError('Either train_mask/test_mask or test_size should be different from None.')
        else:
            check_is_proba(test_size)
            return False
    else:
        check_boolean(train_mask)
        check_boolean(test_mask)
        return True

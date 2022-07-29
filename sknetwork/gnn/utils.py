#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Union, Optional, Tuple
import warnings

import numpy as np
from scipy import sparse

from sknetwork.utils.check import check_is_proba, check_boolean, check_labels, is_square


def filter_mask(mask: np.ndarray, proportion: Optional[float]):
    """Filter a boolean mask so that the proportion of ones does not exceed some target.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask
    proportion : float
        Target proportion of ones.
    Returns
    -------
    mask_filter : np.ndarray
        New boolean mask
    """
    n_ones = sum(mask)
    if n_ones:
        if proportion:
            ratio = proportion * len(mask) / n_ones
            mask[mask] = np.random.random(n_ones) <= ratio
        else:
            mask = np.zeros_like(mask, dtype=bool)
    return mask


def check_existing_masks(labels: np.ndarray, train_mask: Optional[np.ndarray] = None,
                         val_mask: Optional[np.ndarray] = None, test_mask: Optional[np.ndarray] = None,
                         train_size: Optional[float] = None, val_size: Optional[float] = None,
                         test_size: Optional[float] = None) -> Tuple:
    """Check mask parameters and return mask boolean arrays.

    Parameters
    ----------
    labels: np.ndarray
        Label vectors of length :math:`n`, with :math:`n` the number of nodes in `adjacency`. Labels set to `-1`
        will not be considered for training steps.
    train_mask, val_mask, test_mask: np.ndarray
        Boolean arrays indicating whether nodes are in training/validation/test set.
    train_size, val_size, test_size: float
        Proportion of the nodes in the training/validation/test set (between 0 and 1).
        Only used if when corresponding masks are ``None``.

    Returns
    -------
    Tuple containing:
        * ``True`` if training mask is provided
        * training, validation and test masks w.r.t values in `labels`.
    """
    _, _ = check_labels(labels)

    is_negative_labels = labels < 0

    if train_mask is not None:
        check_boolean(train_mask)
        train_mask_filtered = np.logical_and(train_mask, ~is_negative_labels)
        check_mask_similarity(train_mask, train_mask_filtered)
        train_mask = train_mask_filtered
        if test_mask is not None:
            check_boolean(test_mask)
            if val_mask is not None:
                check_boolean(val_mask)
                val_mask_filtered = np.logical_and(val_mask, ~is_negative_labels)
                check_mask_similarity(val_mask, val_mask_filtered)
                val_mask = val_mask_filtered
                if (train_mask & val_mask & test_mask).any():
                    raise ValueError('Masks are overlapping. Please change masks.')
            else:
                val_mask = np.logical_and(~train_mask, ~test_mask)
                val_mask = np.logical_and(val_mask, ~is_negative_labels)
        else:
            if val_mask is None:
                val_mask = filter_mask(~train_mask, val_size)
            val_mask = np.logical_and(val_mask, ~is_negative_labels)
            test_mask = np.logical_and(~train_mask, ~val_mask)
        return True, train_mask, val_mask, test_mask
    else:
        if train_size is None and test_size is None:
            raise ValueError('Either mask parameters or size parameters should be different from None.')
        for size in [train_size, test_size, val_size]:
            if size is not None:
                check_is_proba(size)
        return False, ~is_negative_labels, None, is_negative_labels


def check_mask_similarity(mask_1: np.ndarray, mask_2: np.ndarray):
    """Print warning if two mask arrays are different."""
    if any(mask_1 != mask_2):
        warnings.warn('Nodes with label "-1" are considered in the train set or the validation set.')


def check_early_stopping(early_stopping: bool, val_mask: np.ndarray, patience: int):
    """Check early stopping parameters."""
    if val_mask is None or patience is None or not any(val_mask):
        return False
    else:
        return early_stopping


def check_normalizations(normalizations: Union[str, list]):
    """Check if normalization is known."""
    available_norms = ['left', 'right', 'both']
    if isinstance(normalizations, list):
        for normalization in normalizations:
            if normalization.lower() not in available_norms:
                raise ValueError("Normalization must be 'left', 'right' or 'both'.")
    elif normalizations.lower() not in available_norms:
        raise ValueError("Normalization must be 'left', 'right' or 'both'.")


def check_layers(layers: Union[str, list]):
    """Check if layer types are known."""
    available_layers = ['gcnconv']
    if isinstance(layers, list):
        for layer in layers:
            if layer.lower() not in available_layers:
                raise ValueError("Layer must be \"GCNConv\".")
    else:
        if layers.lower() not in available_layers:
            raise ValueError("Layer must be \"GCNConv\".")


def check_output(n_channels: int, labels: np.ndarray):
    """Check the output of the GNN.

    Parameters
    ----------
    n_channels : int
        Number of output channels
    labels : np.ndarray
        Vector of labels
    """
    n_labels = len(set(labels[labels >= 0]))
    if n_labels > 2 and n_labels > n_channels:
        raise ValueError("The dimension of the output is too small for the number of labels. "
                         "Please check the `dims` parameter of your GNN or the `labels` parameter.")


def get_layers_parameters(dims: Union[int, list], layers: Union[str, list], activations: Union[str, list],
                          use_bias: Union[bool, list], normalizations: Union[str, list],
                          self_loops: Union[bool, list]) -> list:
    """Get the list of layer parameters.

    Parameters
    ----------
    dims :
        Dimensions of layers (in forward direction).
    layers :
        Layer types.
    activations :
        Activations function.
    use_bias :
        ``True`` if a bias vector is added.
    normalizations :
        Normalizations of adjacency matrix.
    self_loops :
        ``True`` if self loops are added. Allowed input are booleans and lists.

    Returns
    -------
    list
        List of layer parameters.
    """

    check_layers(layers)
    check_normalizations(normalizations)

    if not isinstance(dims, list):
        dims = [dims]
    n_layers = len(dims)
    params = [dims, layers, activations, use_bias, normalizations, self_loops]
    for idx, param in enumerate(params[1:]):
        if isinstance(param, list):
            if len(param) != n_layers:
                raise ValueError('The number of parameters must be equal to the number of layers, '
                                 'as given by the length of ``dims``.')
        else:
            params[idx + 1] = [param] * n_layers

    return params


def has_self_loops(input_matrix: sparse.csr_matrix) -> bool:
    """True if the matrix contains self loops."""
    return all(input_matrix.diagonal().astype(bool))


def add_self_loops(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Add self loops to adjacency matrix.

    Parameters
    ----------
    adjacency : sparse.csr_matrix
        Adjacency matrix of the graph.

    Returns
    -------
    sparse.csr_matrix
        Adjacency matrix of the graph with self loops.
    """
    n_row, n_col = adjacency.shape

    if is_square(adjacency):
        adjacency = sparse.diags(np.ones(n_col), format='csr') + adjacency
    else:
        tmp = sparse.eye(n_row)
        tmp.resize(n_row, n_col)
        adjacency += tmp

    return adjacency

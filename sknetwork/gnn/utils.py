#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Union, Optional, Tuple

import numpy as np
from scipy import sparse

from sknetwork.utils.check import check_is_proba, check_boolean, check_labels, is_square


def check_existing_masks(labels: np.ndarray, train_mask: Optional[np.ndarray] = None,
                         val_size: Optional[float] = None) -> Tuple:
    """Check masks parameters and return masks boolean arrays.

    Parameters
    ----------
    labels: np.ndarray
        Label vectors of length :math:`n`, with :math:`n` the number of nodes in `adjacency`. Labels set to `-1`
        will not be considered for training steps.
    train_mask: np.ndarray
        Boolean array indicating whether nodes are in training set.
    val_size: float
        If `train_mask` is `None`, includes this proportion of the nodes in validation set.

    Returns
    -------
    Tuple containing:
    * `True` if training mask is provided
    * training, validation and test masks w.r.t values in `labels`.
    """
    _, _ = check_labels(labels)

    is_negative_labels = labels == -1

    if train_mask is not None:
        check_boolean(train_mask)
        train_mask = np.logical_and(train_mask, ~is_negative_labels)
        val_mask = np.logical_and(~train_mask, ~is_negative_labels)
        return True, train_mask, val_mask, is_negative_labels
    else:
        if val_size is None:
            raise ValueError('Either train_mask or val_size should be different from None.')
        else:
            check_is_proba(val_size)
            return False, ~is_negative_labels, None, is_negative_labels


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


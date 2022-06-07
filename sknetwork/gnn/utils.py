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


def check_norm(norm: Union[str, list]):
    """Check if normalization is known."""
    available_norms = ['both']
    if isinstance(norm, list):
        for n in norm:
            if n.lower() not in available_norms:
                raise ValueError("Layer must be \"Both\".")
    elif norm.lower() not in available_norms:
        raise ValueError("Layer must be \"Both\".")


def check_layer(layer: Union[str, list]):
    """Check if layer type is known."""
    available_layers = ['gcnconv']
    if isinstance(layer, list):
        for lay in layer:
            if lay.lower() not in available_layers:
                raise ValueError("Layer must be \"GCNConv\".")
    else:
        if layer.lower() not in available_layers:
            raise ValueError("Layer must be \"GCNConv\".")


def check_layers_parameters(layer: Union[str, list], n_hidden: Union[int, list], activation: Union[str, list],
                            use_bias: Union[bool, list], norm: Union[str, list], self_loops: Union[bool, list]) -> list:
    """Check whether layer parameters are valid and return a list of layer parameters.

    Parameters
    ----------
    layer :
        Layer type name. Allowed input are strings and lists.
    n_hidden :
        Size of hidden layer. Allowed input are integers and lists.
    activation :
        Name of activation function. Allowed input are strings and lists.
    use_bias :
        `True` if a bias vector is added. Allowed input are booleans and lists.
    norm :
        Normalization kind for adjacency matrix. Allowed input are strings and lists.
    self_loops :
        `True` if self loops are added. Allowed input are booleans and lists.

    Returns
    -------
    list
        List of layer parameters.
    """

    check_layer(layer)
    check_norm(norm)

    params = [layer, n_hidden, activation, use_bias, norm, self_loops]
    for idx, param in enumerate(params[1:]):
        if isinstance(param, list) and not isinstance(params[0], list):
            if len(param) != 1:
                raise ValueError('Number of parameters != number of layers.')
            else:
                params[idx + 1] = param
        elif isinstance(param, list) and isinstance(params[0], list):
            if len(param) != len(params[0]):
                raise ValueError('Number of parameters != number of layers.')
        elif not isinstance(param, list) and isinstance(params[0], list):
            params[idx + 1] = [param] * len(params[0])
        else:
            params[idx + 1] = [param]
    if isinstance(params[0], list):
        params[0] = params[0]
    else:
        params[0] = [params[0]]

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

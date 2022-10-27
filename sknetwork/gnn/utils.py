#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
import inspect
from typing import Union, Optional, Tuple
import warnings

import numpy as np

from sknetwork.gnn.base_activation import BaseActivation, BaseLoss
from sknetwork.gnn.base_layer import BaseLayer
from sknetwork.gnn.layer import get_layer
from sknetwork.gnn.loss import BinaryCrossEntropy, CrossEntropy
from sknetwork.utils.check import check_is_proba, check_boolean, check_labels


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


def check_param(param, length):
    """Check the length of a parameter if a list.
    """
    if not isinstance(param, list):
        param = length * [param]
    elif len(param) != length:
        raise ValueError('The number of parameters must be equal to the number of layers.')
    return param


def check_loss(layer: BaseLayer):
    """Check the length of a parameter if a list.
    """
    if not issubclass(type(layer.activation), BaseLoss):
        raise ValueError('No loss specified for the last layer.')
    if isinstance(layer.activation, CrossEntropy) and layer.out_channels == 1:
        layer.activation = BinaryCrossEntropy()
    return layer.activation


def get_layers(dims: Union[int, list], layer_types: Union[str, BaseLayer, list],
               activations: Union[str, BaseActivation, list], use_bias: Union[bool, list],
               normalizations: Union[str, list], self_embeddings: Union[bool, list], sample_sizes: Union[int, list],
               loss: Union[str, BaseLoss]) -> list:
    """Get the list of layers.

    Parameters
    ----------
    dims :
        Dimensions of layers (in forward direction).
    layer_types :
        Layer types.
    activations :
        Activation functions.
    use_bias :
        ``True`` if a bias vector is added.
    normalizations :
        Normalizations of adjacency matrix.
    self_embeddings :
        ``True`` if self embeddings are added. Allowed input are booleans and lists.
    sample_sizes
        Size of neighborhood sampled for each node.
    loss :
        Loss function.

    Returns
    -------
    list
        List of layers.
    """
    check_normalizations(normalizations)

    if not isinstance(dims, list):
        dims = [dims]
    n_layers = len(dims)

    layer_types = check_param(layer_types, n_layers)
    activations = check_param(activations, n_layers)
    use_bias = check_param(use_bias, n_layers)
    normalizations = check_param(normalizations, n_layers)
    self_embeddings = check_param(self_embeddings, n_layers)
    sample_sizes = check_param(sample_sizes, n_layers)

    layers = []
    names_params = ['layer', 'out_channels', 'activation', 'use_bias', 'normalization', 'self_embeddings',
                    'sample_size']
    for i in range(n_layers):
        params = [layer_types[i], dims[i], activations[i], use_bias[i], normalizations[i], self_embeddings[i],
                  sample_sizes[i]]
        if i == n_layers - 1:
            params.append(loss)
            names_params.append('loss')
        dict_params = dict(zip(names_params, params))
        layers.append(get_layer(**dict_params))

    return layers

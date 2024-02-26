#!/usr/bin/env python3
# coding: utf-8
"""
Created in April 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Iterable, Union

import numpy as np

from sknetwork.gnn.base_activation import BaseActivation, BaseLoss
from sknetwork.gnn.base_layer import BaseLayer
from sknetwork.gnn.layer import get_layer
from sknetwork.gnn.loss import BinaryCrossEntropy, CrossEntropy


def check_early_stopping(early_stopping: bool, val_mask: np.ndarray, patience: int):
    """Check early stopping parameters."""
    if val_mask is None or patience is None or not any(val_mask):
        return False
    else:
        return early_stopping


def check_normalizations(normalizations: Union[str, Iterable]):
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


def get_layers(dims: Union[int, Iterable], layer_types: Union[str, BaseLayer, Iterable],
               activations: Union[str, BaseActivation, list], use_bias: Union[bool, Iterable],
               normalizations: Union[str, Iterable], self_embeddings: Union[bool, Iterable],
               sample_sizes: Union[int, Iterable], loss: Union[str, BaseLoss]) -> list:
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

    if isinstance(dims, int):
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

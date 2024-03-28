#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in July 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Optional, Union

import numpy as np

from sknetwork.gnn.activation import BaseActivation, get_activation
from sknetwork.gnn.loss import BaseLoss, get_loss


class BaseLayer:
    """Base class for GNN layers.

    Parameters
    ----------
    layer_type : str
        Layer type. Can be either ``'Conv'`` (Convolution) or ``'Sage'`` (GraphSAGE).
    out_channels: int
        Dimension of the output.
    activation: str (default = ``'Relu'``) or custom activation.
        Activation function.
        If a string, can be either ``'Identity'``, ``'Relu'``, ``'Sigmoid'`` or ``'Softmax'``.
    use_bias: bool (default = `True`)
        If ``True``, add a bias vector.
    normalization: str (default = ``'both'``)
        Normalization of the adjacency matrix for message passing.
        Can be either `'left'`` (left normalization by the degrees), ``'right'`` (right normalization by the degrees),
        ``'both'`` (symmetric normalization by the square root of degrees, default) or ``None`` (no normalization).
    self_embeddings: bool (default = `True`)
        If ``True``, consider self-embedding in addition to neighbors embedding for each node of the graph.
    sample_size: int (default = 25)
        Size of neighborhood sampled for each node. Used only for ``'SAGEConv'`` layer.

    Attributes
    ----------
    weight : np.ndarray,
        Trainable weight matrix.
    bias : np.ndarray
        Bias vector.
    embedding : np.ndarray
        Embedding of the nodes (before activation).
    output : np.ndarray
        Output of the layer (after activation).
    """
    def __init__(self, layer_type: str, out_channels: int, activation: Optional[Union[BaseActivation, str]] = 'Relu',
                 use_bias: bool = True, normalization: str = 'both', self_embeddings: bool = True,
                 sample_size: int = 25, loss: Optional[Union[BaseLoss, str]] = None):
        self.layer_type = layer_type
        self.out_channels = out_channels
        if loss is None:
            self.activation = get_activation(activation)
        else:
            self.activation = get_loss(loss)
        self.use_bias = use_bias
        self.normalization = normalization.lower()
        self.self_embeddings = self_embeddings
        self.sample_size = sample_size
        self.weight = None
        self.bias = None
        self.embedding = None
        self.output = None
        self.weights_initialized = False

    def _initialize_weights(self, in_channels: int):
        """Initialize weights and bias.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        """
        # He initialization
        self.weight = np.random.randn(in_channels, self.out_channels) * np.sqrt(2 / self.out_channels)
        if self.use_bias:
            self.bias = np.zeros((1, self.out_channels))
        self.weights_initialized = True

    def forward(self, *args, **kwargs):
        """Compute forward pass."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """ String representation of object

        Returns
        -------
        str
            String representation of object
        """
        print_attr = ['out_channels', 'layer_type', 'activation', 'use_bias', 'normalization', 'self_embeddings']
        if 'sage' in self.layer_type:
            print_attr.append('sample_size')
        attributes_dict = {k: v for k, v in self.__dict__.items() if k in print_attr}
        string = ''

        for k, v in attributes_dict.items():
            if k == 'activation':
                string += f'{k}: {v.name}, '
            else:
                string += f'{k}: {v}, '

        return f'  {self.__class__.__name__}({string[:-2]})'

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
import numpy as np


class BaseLayer:
    """Base class for GNN layers.

    Attributes
    ----------
    weight: np.ndarray,
        Trainable weight matrix.
    bias: np.ndarray
        Bias vector.
    update: np.ndarray
        Embedding of the nodes before the activation function.
    embedding: np.ndarray
        Embedding of the nodes after convolution layer.
    """
    def __init__(self, out_channels: int, activation: str = 'Relu', use_bias: bool = True,
                 normalization: str = 'Both', self_loops: bool = True):
        self.out_channels = out_channels
        self.activation = activation.lower()
        self.use_bias = use_bias
        self.normalization = normalization.lower()
        self.self_loops = self_loops
        self.weight = None
        self.bias = None
        self.embedding = None
        self.update = None
        self.weights_initialized = False

    def _initialize_weights(self, in_channels: int):
        """Initialize weights and bias.

        Parameters
        ----------
            in_channels:
        """
        # Trainable parameters with He-et-al initialization
        self.weight = np.random.randn(in_channels, self.out_channels) * np.sqrt(2 / self.out_channels)
        if self.use_bias:
            self.bias = np.zeros((self.out_channels, 1)).T
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
        print_attr = ['out_channels', 'activation', 'use_bias', 'normalizations', 'self_loops']
        attributes_dict = {k: v for k, v in self.__dict__.items() if k in print_attr}
        lines = ''

        for k, v in attributes_dict.items():
            lines += f'{k}: {v}, '

        return f'  {self.__class__.__name__}({lines[:-2]})'

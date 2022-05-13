#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""

import numpy as np
from scipy import sparse

from typing import Union

from sknetwork.linalg import diag_pinv
from sknetwork.gnn.activation import *
from sknetwork.gnn.loss import *
from sknetwork.gnn.utils import check_norm


class GCNConv():
    """ The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    :math:`H^{\prime}=\sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}HW + b)`,

    where :math:`\hat{A} = A + I` denotes the adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii}` its diagonal degree matrix. :math:`W` and :math:`b` are trainable parameters.

    Parameters
    ----------
    in_channel: int
        Size of each input sample.
    out_channel: int
        Size of each output sample.
    use_bias: bool (default=True)
        If True, add a bias vector.
    norm: str (default='both')
        Compute symmetric normalizaztion coefficients.
    self_loops: bool (default=True)
        If True, add self-loops to each node in the graph.
    activation: {'relu', 'tanh', 'sigmoid'}, default='sigmoid'
        Activation function name:

        - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function, returns f(x) = max(0, x)

    Attributes
    ----------
    W, b: np.ndarray, np.ndarray
        Trainable weight matrix and bias vector.
    Z: np.ndarray
        :math:`Z=AHW + b` with :math:`A` the adjacency matrix of the graph, :math:`H` the feature matrix of the graph,
        :math:`W` the trainable weight matrix and :math:`b` the bias vector (if needed).
    emb: np.ndarray
        Embedding of the nodes after convolution layer.

    Returns
    -------
    np.ndarray
        Nodes embedding.

    """

    def __init__(self, in_channels: int, out_channels: int, use_bias: bool = True, norm: str = 'both',
                 self_loops: bool = True, activation: str = 'sigmoid'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        check_norm(norm)
        self.norm = norm
        self.self_loops = self_loops
        self.activation = activation

        # Weight matrix with He-et-al initialization
        self.W = np.random.randn(in_channels, out_channels) * np.sqrt(2 / out_channels)
        if self.use_bias:
            self.bias = np.zeros((out_channels, 1)).T

    def forward(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
                feat: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Compute graph convolution.

        Parameters
        ----------
        adjacency
            Adjacency matrix of the graph.
        feat : sparse.csr_matrix, np.ndarray
            Input feature of shape (n, d) with n the number of nodes in the graph and d the size of feature space.

        Returns
        -------
        np.ndarray
            Nodes embedding.
        """

        n_col = adjacency.shape[1]

        if self.self_loops:
            adjacency = sparse.diags(np.ones(n_col), format='csr') + adjacency

        if self.norm == 'both':
            weights = adjacency.dot(np.ones(n_col))
            d_inv = diag_pinv(np.sqrt(weights))
            adjacency = d_inv.dot(adjacency).dot(d_inv)

        activation_function = ACTIVATIONS[self.activation]
        msg = adjacency.dot(feat)

        Z = msg.dot(self.W)
        if self.use_bias:
            Z += self.bias

        emb = activation_function(Z)

        # Keep track of results for backprop
        self.emb = emb
        self.Z = Z

        return self.emb

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """ String representation of object

        Returns
        -------
        str
            String representation of object
        """
        attributes_dict = {k: v for k, v in self.__dict__.items() if k not in ['W', 'bias']}
        lines = ''

        for k, v in attributes_dict.items():
            lines += f'{k}: {v}, '

        return f'  {self.__class__.__name__}({lines[:-2]})'
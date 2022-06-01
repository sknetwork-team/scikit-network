#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.gnn.activation import get_activation_function
from sknetwork.gnn.utils import check_norm
from sknetwork.linalg import diag_pinv


class GCNConv:
    """Graph convolutional operator.

    :math:`H^{\prime}=\sigma(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}HW + b)`,

    where :math:`\hat{A} = A + I` denotes the adjacency matrix with inserted self-loops and
    :math:`\hat{D}` its diagonal degree matrix. :math:`W` and :math:`b` are trainable parameters.

    Parameters
    ----------
    in_channels: int
        Size of each input sample.
    out_channels: int
        Size of each output sample.
    use_bias: bool (default=True)
        If True, add a bias vector.
    norm: str (default='both')
        Normalization kind for adjacency matrix.
        - 'both', computes symmetric normalization
    self_loops: bool (default=True)
        If True, add self-loops to each node in the graph.
    activation: {'Relu', 'Softmax', 'Sigmoid'}, default='Sigmoid'
        Activation function name:

        - 'Relu', the rectified linear unit function, returns f(x) = max(0, x)
        - 'Sigmoid', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).

    Attributes
    ----------
    W, bias: np.ndarray, np.ndarray
        Trainable weight matrix and bias vector.
    Z: np.ndarray
        :math:`Z=AHW + b` with :math:`A` the adjacency matrix of the graph, :math:`H` the feature matrix of the graph,
        :math:`W` the trainable weight matrix and :math:`b` the bias vector (if needed).
    emb: np.ndarray
        Embedding of the nodes after convolution layer.

    References
    ----------
    Kipf, T., & Welling, M. (2017).
    `Semi-supervised Classification with Graph Convolutional Networks.
    <https://arxiv.org/pdf/1609.02907.pdf>`_
    5th International Conference on Learning Representations.
    """

    def __init__(self, in_channels: int, out_channels: int, use_bias: bool = True, norm: str = 'both',
                 self_loops: bool = True, activation: str = 'Sigmoid'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        check_norm(norm)
        self.norm = norm
        self.self_loops = self_loops
        self.activation = activation.lower()

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

        activation_function = get_activation_function(self.activation)
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

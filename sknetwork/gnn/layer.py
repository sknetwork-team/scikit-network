#!/usr/bin/env python3
# coding: utf-8
"""
Created on Thu Apr 21 2022
@author: Simon Delarue <sdelarue@enst.fr>
"""
from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.gnn.activation import BaseActivation
from sknetwork.gnn.loss import BaseLoss
from sknetwork.gnn.base_layer import BaseLayer
from sknetwork.gnn.neigh_sampler import UniformNeighborSampler
from sknetwork.utils.check import has_self_loops, add_self_loops
from sknetwork.linalg import diag_pinv


class Convolution(BaseLayer):
    """Graph convolutional layer.

    Apply the following function to the embedding :math:`X`:

    :math:`\\sigma(\\bar AXW + b)`,

    where :math:`\\bar A` is the normalized adjacency matrix (possibly with inserted self-loops),
    :math:`W`, :math:`b` are trainable parameters and :math:`\\sigma` is the activation function.

    Parameters
    ----------
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
    self_loops: bool (default = `True`)
        If ``True``, add a self-loop of unit weight to each node of the graph.
    sample_size: int (default = 25)
        Size of neighborhood sampled for each node. Used only for ``'SAGEConv'`` layer.

    Attributes
    ----------
    weight: np.ndarray,
        Trainable weight matrix.
    bias: np.ndarray
        Bias vector.
    embedding: np.ndarray
        Embedding of the nodes (before activation).
    output: np.ndarray
        Output of the layer (after activation).

    References
    ----------
    Kipf, T., & Welling, M. (2017).
    `Semi-supervised Classification with Graph Convolutional Networks.
    <https://arxiv.org/pdf/1609.02907.pdf>`_
    5th International Conference on Learning Representations.
    """
    def __init__(self, layer_type: str, out_channels: int, activation: Optional[Union[BaseActivation, str]] = 'Relu',
                 use_bias: bool = True, normalization: str = 'both', self_loops: bool = True, sample_size: int = None,
                 loss: Optional[Union[BaseLoss, str]] = None):
        super(Convolution, self).__init__(layer_type, out_channels, activation, use_bias, normalization, self_loops,
                                          sample_size, loss)

    def forward(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
                features: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Compute graph convolution.

        Parameters
        ----------
        adjacency
            Adjacency matrix of the graph.
        features : sparse.csr_matrix, np.ndarray
            Input feature of shape :math:`(n, d)` with :math:`n` the number of nodes in the graph and :math:`d`
            the size of feature space.

        Returns
        -------
        output: np.ndarray
            Output of the layer.
        """
        if not self.weights_initialized:
            self._initialize_weights(features.shape[1])

        n_row, n_col = adjacency.shape

        # Neighbor sampling
        if self.layer_type == 'sageconv':
            sampler = UniformNeighborSampler(sample_size=self.sample_size)
            adjacency = sampler(adjacency)

        if self.self_loops and self.layer_type == 'conv':
            if not has_self_loops(adjacency):
                adjacency = add_self_loops(adjacency)

        weights = adjacency.dot(np.ones(n_col))
        if self.normalization == 'left':
            d_inv = diag_pinv(weights)
            adjacency = d_inv.dot(adjacency)
        elif self.normalization == 'right':
            d_inv = diag_pinv(weights)
            adjacency = adjacency.dot(d_inv)
        elif self.normalization == 'both':
            d_inv = diag_pinv(np.sqrt(weights))
            adjacency = d_inv.dot(adjacency).dot(d_inv)

        if self.self_loops and self.layer_type == 'sageconv':
            if not has_self_loops(adjacency):
                adjacency = add_self_loops(adjacency)

        message = adjacency.dot(features)
        embedding = message.dot(self.weight)

        if self.use_bias:
            embedding += self.bias

        output = self.activation.output(embedding)

        self.embedding = embedding
        self.output = output

        return output


def get_layer(layer: Union[BaseLayer, str] = 'conv', *args) -> BaseLayer:
    """Get layer.

    Parameters
    ----------
    layer : str or custom layer
        If a string, must be ``'Conv'``.

    Returns
    -------
    Layer object.
    """
    if issubclass(type(layer), BaseLayer):
        return layer
    elif type(layer) == str:
        layer = layer.lower()
        if layer in ['conv', 'gcnconv', 'graphconv']:
            return Convolution(layer, *args)
        elif layer in ['sage', 'sageconv']:
            params = [arg for arg in args]
            params[3] = 'left'
            params[4] = True
            return Convolution(layer, *params)
        else:
            raise ValueError("Layer name must be \"Conv\" or \"SAGEConv\".")
    else:
        raise TypeError("Layer must be a string or a \"BaseLayer\" object.")

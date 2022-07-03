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
from sknetwork.gnn.utils import has_self_loops, add_self_loops
from sknetwork.linalg import diag_pinv


class GCNConv:
    """Graph convolutional operator.

    :math:`H^{\prime}=\sigma(D^{-1/2}\hat{A}D^{-1/2}HW + b)`,

    where :math:`\hat{A} = A + I` denotes the adjacency matrix with inserted self-loops and
    :math:`D` its diagonal degree matrix. :math:`W` and :math:`b` are trainable parameters.

    Parameters
    ----------
    out_channels: int
        Size of each output sample.
    activation: str (default = ``'Relu'``)
        Activation function name:

        * ``'Relu'``, the rectified linear unit function, returns f(x) = max(0, x)
        * ``'Sigmoid'``, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
        * ``'Softmax'``, the softmax function, returns f(x) = exp(x) / sum(exp(x))
    use_bias: bool (default = `True`)
        If ``True``, add a bias vector.
    norm: str (default = ``'Both'``)
        Normalization kind for adjacency matrix.
        * ``'Both'``, computes symmetric normalization
    self_loops: bool (default = `True`)
        If ``True``, add self-loops to each node in the graph.

    Attributes
    ----------
    weight, bias: np.ndarray, np.ndarray
        Trainable weight matrix and bias vector.
    update: np.ndarray
        :math:`\text{update}=AHW + b` with :math:`A` the adjacency matrix of the graph, :math:`H` the feature matrix
        of the graph, :math:`W` the trainable weight matrix and :math:`b` the bias vector (if needed).
    emb: np.ndarray
        Embedding of the nodes after convolution layer: :math:`\text{emb}=\sigma(AHW + b)`,
        with :math:`\sigma` an activation function.

    References
    ----------
    Kipf, T., & Welling, M. (2017).
    `Semi-supervised Classification with Graph Convolutional Networks.
    <https://arxiv.org/pdf/1609.02907.pdf>`_
    5th International Conference on Learning Representations.

    He, K. & Zhang, X. & Ren, S. & Sun, J. (2015).
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.
    <https://arxiv.org/pdf/1502.01852.pdf>`_
    Proceedings of the IEEE International Conference on Computer Vision (ICCV).
    """

    def __init__(self, out_channels: int, activation: str = 'Relu', use_bias: bool = True, norm: str = 'Both',
                 self_loops: bool = True):
        self.out_channels = out_channels
        self.activation = activation.lower()
        self.use_bias = use_bias
        self.norm = norm.lower()
        self.self_loops = self_loops
        self.weight = None
        self.bias = None
        self.emb = None
        self.update = None
        self.weights_initialized = False

    def _initialize_weights(self, in_channels: int):
        # Trainable parameters with He-et-al initialization
        self.weight = np.random.randn(in_channels, self.out_channels) * np.sqrt(2 / self.out_channels)
        if self.use_bias:
            self.bias = np.zeros((self.out_channels, 1)).T
        self.weights_initialized = True

    def forward(self, adjacency: Union[sparse.csr_matrix, np.ndarray],
                feat: Union[sparse.csr_matrix, np.ndarray]) -> np.ndarray:
        """Compute graph convolution.

        Parameters
        ----------
        adjacency
            Adjacency matrix of the graph.
        feat : sparse.csr_matrix, np.ndarray
            Input feature of shape :math:`(n, d)` with :math:`n` the number of nodes in the graph and :math:`d`
            the size of feature space.

        Returns
        -------
        np.ndarray
            Nodes embedding.
        """
        if not self.weights_initialized:
            self._initialize_weights(feat.shape[1])

        n_row, n_col = adjacency.shape

        if self.self_loops:
            if not has_self_loops(adjacency):
                adjacency = add_self_loops(adjacency)

        if self.norm == 'both':
            weights = adjacency.dot(np.ones(n_col))
            d_inv = diag_pinv(np.sqrt(weights))
            adjacency = d_inv.dot(adjacency).dot(d_inv)

        msg = adjacency.dot(feat)
        update = msg.dot(self.weight)

        if self.use_bias:
            update += self.bias

        activation_function = get_activation_function(self.activation)
        emb = activation_function(update)

        # Keep track of results for backprop
        self.emb = emb
        self.update = update

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
        print_attr = ['out_channels', 'activation', 'use_bias', 'norm', 'self_loops']
        attributes_dict = {k: v for k, v in self.__dict__.items() if k in print_attr}
        lines = ''

        for k, v in attributes_dict.items():
            lines += f'{k}: {v}, '

        return f'  {self.__class__.__name__}({lines[:-2]})'


def get_layer(layer: str = 'GCNConv', *args) -> object:
    """Instantiate layer according to parameters.

    Parameters
    ----------
    layer : str
        Which layer to use. Can be ``'GCNConv'``.

    Returns
    -------
    Layer object.
    """
    layer = layer.lower()
    if layer == 'gcnconv':
        return GCNConv(*args)
    else:
        raise ValueError("Layer must be \"GCNConv\".")

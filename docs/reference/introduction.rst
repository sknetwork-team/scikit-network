.. _introduction:

Introduction
************

scikit-network is an open-source python package for the analysis of large graphs.
Graphs are represented by their adjacency matrix, in one of the following two formats:

* numpy array
* scipy sparse csr matrix

Quick start
-----------

Install scikit-network:

.. code-block:: console

    $ pip install scikit-network

To use scikit-network in a project::

    import sknetwork as skn


About the documentation
-----------------------

We use the following notations in the documentation:

**Undirected graphs**

* :math:`A` is the adjacency matrix of the graph.
* :math:`d = A1` is the vector of node weights (node degrees if the matrix :math:`A` is binary)
* :math:`D = \text{diag}(d)` the diagonal matrix of node weights.
* :math:`L = D - A` is the Laplacian matrix of the graph.
* :math:`P = D^{-1}A` is the transition matrix of the random walk in the graph (for positive node weights).
* :math:`w = 1^T A1` is the total weight of the graph.

**Directed graphs**

* :math:`A` is the adjacency matrix of the graph.
* :math:`d^+ = A1` is the vector of node out-weights (node out-degrees if the matrix :math:`A` is binary)
* :math:`D^+ = \text{diag}(d^+)` is the diagonal matrix of node out-weights.
* :math:`P^+= {D^+}^{-1}A` is the transition matrix of the random walk in the graph (for positive node out-weights).
* :math:`d^- = A1` is the vector of node in-weights (node in-degrees if the matrix :math:`A` is binary)
* :math:`D^- = \text{diag}(d^-)` is the diagonal matrix of node in-weights.
* :math:`P^-= {D^-}^{-1}A` is the transition matrix of the random walk in reverse direction  (for positive node in-weights).
* :math:`w = 1^T A1` is the total weight of the graph.

**Bipartite graphs

* :math:`B` denotes the biadjacency matrix of the graph, rows and columns corresponding to samples and features.
* :math:`d = B1` is the vector of sample weights.
* :math:`D = \text{diag}(d)` is the diagonal matrix of sample weights.
* :math:`f = B^T1` is the vector of feature weights.
* :math:`F = \text{diag}(f)` the diagonal matrix of feature weights.
* :math:`w = 1^T B1` is the total weight of the graph.


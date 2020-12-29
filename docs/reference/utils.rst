.. _utils:

Utils
*****

Various tools for graph analysis.

Build graphs
------------

.. autofunction:: sknetwork.utils.edgelist2adjacency

.. autofunction:: sknetwork.utils.edgelist2biadjacency


Convert graphs
--------------

.. autofunction:: sknetwork.utils.bipartite2directed

.. autofunction:: sknetwork.utils.bipartite2undirected

.. autofunction:: sknetwork.utils.directed2undirected


Get neighbors
-------------

.. autofunction:: sknetwork.utils.get_neighbors


Clustering
----------

.. autofunction:: sknetwork.utils.membership_matrix

.. autoclass:: sknetwork.utils.KMeansDense

.. autoclass:: sknetwork.utils.WardDense


Nearest-neighbors
-----------------

.. autoclass:: sknetwork.utils.KNNDense

.. autoclass:: sknetwork.utils.CNNDense


Co-Neighborhood
---------------

.. autofunction:: sknetwork.utils.co_neighbor_graph


Projection
----------

.. autofunction:: sknetwork.utils.projection_simplex

.. autofunction:: sknetwork.utils.projection_simplex_array

.. autofunction:: sknetwork.utils.projection_simplex_csr


.. _utils:

Utils
*****

Various tools for graph analysis.


Convert graphs
--------------

.. autofunction:: sknetwork.utils.directed2undirected

.. autofunction:: sknetwork.utils.bipartite2undirected

.. autofunction:: sknetwork.utils.bipartite2directed


Neighborhood
------------

.. autofunction:: sknetwork.utils.get_degrees

.. autofunction:: sknetwork.utils.get_weights

.. autofunction:: sknetwork.utils.get_neighbors


Clustering
----------

.. autoclass:: sknetwork.utils.KMeansDense

.. autoclass:: sknetwork.utils.WardDense

Membership matrix
-----------------

.. autofunction:: sknetwork.utils.get_membership

.. autofunction:: sknetwork.utils.from_membership


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


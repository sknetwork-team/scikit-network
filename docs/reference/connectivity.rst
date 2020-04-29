.. _connectivity:

Connectivity
************

This module contains standard algorithms related to graph connectivity: shortest paths, graph traversals, etc.

Most algorithms are adapted from SciPy_.

Shortest paths
--------------

.. autofunction:: sknetwork.connectivity.shortest_path

Summary of the different methods and their worst-case complexity for :math:`n` nodes and :math:`m` edges (for the
all-pairs problem):

+----------------+------------------------------+----------------------------------------+
|     Method     |  Worst-case time complexity  |                 Remarks                |
+================+==============================+========================================+
| Floyd-Warshall | :math:`O(n^3)`               | All-pairs shortest path problem only   |
+----------------+------------------------------+----------------------------------------+
|    Dijkstra    | :math:`O(n^2 \log n + nm)`   | For use on graphs                      |
|                |                              | with positive weights only             |
+----------------+------------------------------+----------------------------------------+
|  Bellman-Ford  | :math:`O(nm)`                |                                        |
+----------------+------------------------------+ For use on graphs without              |
|    Johnson     | :math:`O(n^2 \log n + nm)`   | negative-weight cycles only            |
+----------------+------------------------------+----------------------------------------+


Searches
--------

.. autofunction:: sknetwork.connectivity.breadth_first_search

.. autofunction:: sknetwork.connectivity.depth_first_search



Structure
---------

.. autofunction:: sknetwork.connectivity.connected_components

.. autofunction:: sknetwork.connectivity.largest_connected_component

.. autofunction:: sknetwork.connectivity.is_bipartite


.. _SciPy: https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html

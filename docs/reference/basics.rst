.. _basics:

Basics
******

This module contains standard algorithms for shortest path computation, structure of the graph and searches (BFS, DFS).

Most algorithms are adapted from SciPy_.

Shortest paths
--------------

.. autofunction:: sknetwork.basics.shortest_path

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

.. autofunction:: sknetwork.basics.breadth_first_search

.. autofunction:: sknetwork.basics.depth_first_search



Structure
---------

.. autofunction:: sknetwork.basics.is_connected

.. autofunction:: sknetwork.basics.connected_components

.. autofunction:: sknetwork.basics.largest_connected_component

.. autofunction:: sknetwork.basics.is_bipartite


Co-Neighborhood
---------------

.. autofunction:: sknetwork.basics.co_neighbors_graph

.. autoclass:: sknetwork.basics.CoNeighbors


.. _SciPy: https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html

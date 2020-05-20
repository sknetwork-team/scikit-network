.. _paths:

Paths
*****

Standard algorithms related to graph traversal.

Most algorithms are adapted from SciPy_.

Shortest paths
--------------

.. autofunction:: sknetwork.paths.distance

.. autofunction:: sknetwork.paths.shortest_path

Summary of the different methods and their worst-case complexity for :math:`n` nodes and :math:`m` edges (for the
all-pairs problem):

+----------------+------------------------------+----------------------------------------+
|     Method     |  Worst-case time complexity  |                 Remarks                |
+================+==============================+========================================+
|    Dijkstra    | :math:`O(n^2 \log n + nm)`   | For use on graphs                      |
|                |                              | with positive weights only             |
+----------------+------------------------------+----------------------------------------+
|  Bellman-Ford  | :math:`O(nm)`                |                                        |
+----------------+------------------------------+ For use on graphs without              |
|    Johnson     | :math:`O(n^2 \log n + nm)`   | negative-weight cycles only            |
+----------------+------------------------------+----------------------------------------+


Searches
--------

.. autofunction:: sknetwork.paths.breadth_first_search

.. autofunction:: sknetwork.paths.depth_first_search



.. _SciPy: https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html

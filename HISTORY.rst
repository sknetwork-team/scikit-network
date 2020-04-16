=======
History
=======

0.13.3 (2020-04-13)
-------------------

* Minor bug

0.13.2 (2020-04-13)
-------------------

* Added wheels for multiple platforms (OSX, Windows (32 & 64 bits) and many Linux) and Python version (3.6/3.7/3.8)
* Documentation update (SVG dendrograms, tutorial updates)

0.13.1a (2020-04-09)
--------------------

* Minor bug

0.13.0a (2020-04-09)
--------------------

* Changed from Numba to Cython for better performance
* Added visualization module
* Added k-nearest neighbors classifier
* Added Louvain hierarchy
* Added predict method in embedding
* Added soft clustering to clustering algorithms
* Added soft classification to classification algorithms
* Added graphs in data module
* Various API change

0.12.1 (2020-01-20)
-------------------

* Added heat kernel based node classifier
* Updated loaders for WikiLinks
* Fixed file-related issues for Windows

0.12.0 (2019-12-10)
-------------------

* Added VerboseMixin for verbosity features
* Added Loaders for WikiLinks & Konect databases

0.11.0 (2019-11-28)
-------------------

* sknetwork: new API for bipartite graphs
* new module: Soft node classification
* new module: Node classification
* new module: data (merge toy graphs + loader)
* clustering: Spectral Clustering
* ranking: new algorithms
* utils: K-neighbors
* hierarchy: Spectral WardDense
* data: loader (Vital Wikipedia)

0.10.1 (2019-08-26)
-------------------

* Minor bug

0.10.0 (2019-08-26)
-------------------

* Clustering (and related metrics) for directed and bipartite graphs
* Hierarchical clustering (and related metrics) for directed and bipartite graphs
* Fix bugs on embedding algorithms


0.9.0 (2019-07-24)
------------------

* Change parser output
* Fix bugs in ranking algorithms (zero-degree nodes)
* Add notebooks
* Import algorithms from scipy (shortest path, connected components, bfs/dfs)
* Change SVD embedding (now in decreasing order of singular values)

0.8.2 (2019-07-19)
------------------

* Minor bug

0.8.1 (2019-07-18)
------------------

* Added diffusion ranking
* Minor fixes
* Minor doc tweaking

0.8.0 (2019-07-17)
------------------

* Changed Louvain, BiLouvain, Paris and PageRank APIs
* Changed PageRank method
* Documentation overhaul
* Improved Jupyter tutorials

0.7.1 (2019-07-04)
------------------

* Added Algorithm class for nicer repr of some classes
* Added Jupyter notebooks as tutorials in the docs
* Minor fixes

0.7.0 (2019-06-24)
------------------

* Updated PageRank
* Added tests for Numba versioning

0.6.1 (2019-06-19)
------------------

* Minor bug

0.6.0 (2019-06-19)
------------------

* Largest connected component
* Simplex projection
* Sparse Low Rank Decomposition
* Numba support for Paris
* Various fixes and updates

0.5.0 (2019-04-18)
------------------

* Unified Louvain.

0.4.0 (2019-04-03)
------------------

* Added Louvain for directed graphs and ComboLouvain for bipartite graphs.

0.3.0 (2019-03-29)
------------------

* Updated clustering module and documentation.

0.2.0 (2019-03-21)
------------------

* First real release on PyPI.

0.1.1 (2018-05-29)
------------------

* First release on PyPI.

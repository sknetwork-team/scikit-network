=======
History
=======

0.33.3 (2025-05-15)
-------------------

* Fix wheel generation for Python 3.13

0.33.2 (2025-05-12)
-------------------

* Add Python 3.13
* Add wheel for aarch64
* Fix bug on color display

0.33.1 (2024-09-18)
-------------------

* Upgrade Numpy

0.33.0 (2024-07-16)
-------------------

* Add Python 3.12
* Drop Python 3.8

0.32.1 (2024-04-02)
-------------------

* Fix documentation
* Fix wheel upload

0.32.0 (2024-03-29)
-------------------

* Add Leiden clustering algorithm
* Add k-center clustering algorithm
* Add functions to detect and break cycles
* Add damping factor in diffusion
* Fix F1 scores
* Remove hierarchical Louvain embedding
* Get clustering coefficient for directed graphs

0.31.0 (2023-05-22)
-------------------

* Add Python 3.11
* Add set_param / get_param to algorithms, suggested by Franz Kiraly (#557)
* Compute shortest paths by matrix-vector multiplications
* Make tools on topology (cliques, code-decomposition, etc.) as functions
* Rename parameter membership -> probs for soft classification / clustering
* Add softmax to classification by diffusion

0.30.0 (2023-04-12)
-------------------

* Add overview
* Add predict_proba method to classification and clustering

0.29.0 (2023-03-30)
-------------------

* Change API for clustering (predict / transform)
* Change API for classification (seeds -> labels)
* Change API for ranking (seeds -> weights)
* Change API for GNN (same as classifiers)
* Remove first order methods for link prediction
* Add nearest neighbor methods for link prediction
* Add KNN classifier without embedding
* Add TF-IDF
* Solve security issues by upgrade of wheels and ipython

0.28.3 (2023-01-06)
-------------------

* Drop Python 3.7
* Update Numpy requirement

0.28.2 (2022-11-30)
-------------------

* Allow Python 3.7, by Thomas Bonald
* Fix input format for KMeans, issue #548 raised by @sgerbe

0.28.1 (2022-11-22)
-------------------

* Fix sampling for GraphSage, by Simon Delarue
* Fix leakage on GNNs, by Thomas Bonald and Simon Delarue
* Update tutorial on GNNs with node inference, by Thomas Bonald and Simon Delarue

0.28.0 (2022-11-16)
-------------------

* Update Graph neural networks (e.g., add GraphSAGE), by Simon Delarue
* Clean data home folder (set one folder per dataset collection, NetSet, Konect, ...), by Thomas Bonald
* Improve classification by diffusion, setting -1 to unreached nodes, by Thomas Bonald
* Fix bug on modularity, raised by Alessandro (#543)
* Clean up source distribution, by Nicholas Bollweg (#544)
* Safe file extraction, by TrellixVulnTeam
* Fix bug on dendrogram cut, raised by Nina Sachdev (#546)
* Add a function to aggregate a graph per label, by Thomas Bonald

0.27.1 (2022-07-29)
-------------------

* Fix documentation

0.27.0 (2022-07-29)
-------------------

* Drop Python 3.7
* Update NumPy and SciPy requirements
* Add graph neural networks, by Simon Delarue (#533)
* Add fit_predict / fit_transform where appropriate, by Thomas Bonald
* Add Louvain hierarchical clustering (bottom-up), by Thomas Bonald
* Improve classification by diffusion (vectorial), by Thomas Bonald
* Add F1 scores for classification, by Thomas Bonald
* Add cosine similarity metric for embeddings, by Thomas Bonald
* Add acyclic test for undirected graphs, by Thomas Bonald
* Update algorithms to accept all sparse matrix formats of scipy, by Thomas Bonald

0.26.0 (2022-05-03)
-------------------

* Add module on regression, by Thomas Bonald
* Add connected components for bipartite graphs, by Thomas Bonald
* Update functions for loading graphs, by Thomas Bonald
* Fix shuffling nodes in Louvain (issue #521), by Thomas Bonald
* Add radius and eccentricity metrics, by Henry Carscadden (#522)
* Add new use case (recommendation), by Thomas Bonald

0.25.0 (2022-03-15)
-------------------

* Add use cases as notebooks, by Thomas Bonald
* Add list/dict of neighbors for building graphs, by Thomas Bonald
* Update Spectral embedding, by Thomas Bonald
* Update Block models, by Thomas Bonald (#507)
* Fix Tree sampling divergence, by Thomas Bonald (#505)
* Allow parsers to return weighted graphs, by Thomas Bonald
* Add Apple Silicon and Python 3.10 wheels, by Quentin Lutz (#503)

0.24.0 (2021-07-27)
-------------------

* Merge Bi* algorithms (e.g., BiLouvain -> Louvain) by Thomas Bonald (#490)
* Transition from Travis to Github actions by Quentin Lutz (#488)
* Added sdist build for conda recipes
* Added name position for graph visualization
* Removed randomized algorithms

0.23.1 (2021-04-24)
-------------------

* Updated NumPy and SciPy requirements

0.23.0 (2021-04-23)
-------------------

* New push-based implementation of PageRank by Wenzhuo Zhao (#475)
* Fixed cut_balanced in hierarchy
* Dropped Python 3.6, wheels for Python 3.9 (switched to manylinux2014)

0.22.0 (2021-02-09)
-------------------

* Added hierarchical Louvain embedding by Quentin Lutz (#468)
* Doc fixes and updates
* Requirements update

0.21.0 (2021-01-29)
-------------------

* Added random projection embedding by Thomas Bonald (#461)
* Added PCA-based embedding by Thomas Bonald (#461)
* Added 64-bit support for Louvain by Flávio Juvenal (#450)
* Added verbosity options for dataset loaders
* Fixed Louvain embedding
* Various doc and tutorial updates

0.20.0 (2020-10-20)
-------------------

* Added betweenness algorithm by Tiphaine Viard (#444)

0.19.3 (2020-09-17)
-------------------

* Added Louvain-based embedding
* Fix documentation with new dataset website URLs

0.19.2 (2020-09-14)
-------------------

* Fix documentation with new dataset website URLs.

0.19.1 (2020-09-09)
-------------------

* Fix visualization features
* Fix documentation

0.19.0 (2020-09-02)
-------------------

* Added link prediction module
* Added pie-node visualization of memberships
* Added Weisfeiler-Lehman graph coloring by Pierre Pebereau and Alexis Barreaux (#394)
* Added Force Atlas 2 graph layout by Victor Manach and Rémi Jaylet (#396)
* Added triangle listing algorithm for directed and undirected graph by Julien Simonnet and Yohann Robert (#376)
* Added k-core decomposition algorithm by Julien Simonnet and Yohann Robert (#377)
* Added k-clique listing algorithm by Julien Simonnet and Yohann Robert (#377)
* Added color map option in visualization module
* Updated NetSet URL

0.18.0 (2020-06-08)
-------------------

* Added Katz centrality
* Refactor connectivity module into paths and topology
* Refactor Diffusion into Dirichlet
* Added parsers for adjacency list TSV and GraphML
* Added shortest paths and distances

0.17.0 (2020-05-07)
-------------------

* Add clustering by label propagation
* Add models
* Add function to build graph from edge list
* Change a parameter in SVG visualization functions
* Minor corrections

0.16.0 (2020-04-30)
-------------------

* Refactor basics module into connectivity
* Cython version for label propagation
* Minor corrections

0.15.2 (2020-04-24)
-------------------

* Clarified requirements
* Minor corrections

0.15.1 (2020-04-21)
-------------------

* Added OpenMP support for all platforms

0.15.0 (2020-04-20)
-------------------

* Updated ranking module : new pagerank solver, new HITS params, post-processing
* Polynomes in linear algebra
* Added meta.name attribute for Bunch
* Minor corrections

0.14.0 (2020-04-17)
-------------------

* Added spring layout in embedding
* Added label propagation in classification
* Added save / load functions in data
* Added display edges parameter in svg graph exports
* Corrected typos in documentation

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

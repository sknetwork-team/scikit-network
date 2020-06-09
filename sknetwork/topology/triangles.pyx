# distutils: language = c++
# cython: language_level=3
"""
Created on Jun 3, 2020
@author: Julien Simonnet <julien.simonnet@etu.upmc.fr>
@author: Yohann Robert <yohann.robert@etu.upmc.fr>
"""

import numpy as np
cimport numpy as np
from scipy import sparse
from cython.parallel import prange, parallel

# from libc.stdio cimport printf

cimport cython

ctypedef np.int_t int_type_t
ctypedef np.uint8_t bool_type_t

@cython.boundscheck(False)
@cython.wraparound(False)

cdef np.ndarray[int, ndim=1] bucketSort(int n_nodes, int[:] indptr):
	"""	   Orders the nodes by their degre using a bucket sort method.

	Parameters
	----------
	n_nodes :
		Number of nodes.
	indptr :
		CSR format index array of the normalized adjacency matrix.
		
	Returns
	-------
	ordered :
		Numpy array of nodes ordered by their degree
	"""
	cdef int[:] cd			# cummulative degrees of the nodes
	cdef np.ndarray[int, ndim=1] ordered		# array of ordered nodes
	
	cdef int i
	
	cd = np.zeros((n_nodes+1,), dtype=np.int32)			# initializes the array to be filled with 0
	ordered = np.empty((n_nodes,), dtype=np.int32)		# initializes the array
	
	cd[0] = 0
	for i in range(n_nodes):	# puts in the array the number of nodes of each specific degree
		cd[(indptr[i+1] - indptr[i]) + 1] += 1
	
	for i in range(n_nodes):	# calculates the cummulative degrees
		cd[i+1] += cd[i]
		
	cdef int d
	for i in range(n_nodes):	# orderers the nodes and puts them at the right place thanks to the cd array
		d = indptr[i+1] - indptr[i]
		ordered[cd[d]] = i
		cd[d] += 1
		
	return ordered
	
cdef long triangles_c(int n_nodes, int[:] indptr, int[:] indices, int[:] indexation):
	"""	   Counts the number of triangles in a DAG.

	Parameters
	----------
	n_nodes :
		Number of nodes.
	indices :
		CSR format index array of the normalized adjacency matrix of a DAG.
	indptr :
		CSR format index pointer array of the normalized adjacency matrix of a DAG.
	indexation:
		Array that associates a node to its position in the order (it is not necessary to count but will be to list)
	printing:
		If ``True`` will print triangles	
	
	Returns
	-------
	nb_triangles :
		Number of triangles in the graph
	"""
	
	cdef int i, j, k
	cdef int u, v
	cdef long nb_triangles = 0		# number of triangles in the DAG
	
	for u in range(n_nodes):
		for k in range(indptr[u], indptr[u+1]):
			v = indices[k]
			i = indptr[u]
			j = indptr[v]
			
			# calculate the intersection of neighbors of u and v
			while (i < indptr[u+1]) and (j < indptr[v+1]):
				if indices[i] == indices[j]:
					# if (printing):
					#	printf("(%d,%d,%d)\n",indexation[u], indexation[v], indexation[i])
						
					i += 1
					j += 1
					nb_triangles += 1	# increments the number of triangles
				else :
					if indices[i] < indices[j]:
						i += 1
					else :
						j += 1
			
	return nb_triangles
	
	
cdef long tri_intersection(int u, int[:] indptr, int[:] indices, int[:] indexation) nogil:
	"""	   Counts the number of nodes in the intersection of a node and its neighbors in a DAG.

	Parameters
	----------
	u :
		Index of the node to study.
	indptr :
		CSR format index pointer array of the normalized adjacency matrix of a DAG.
	indices :
		CSR format index array of the normalized adjacency matrix of a DAG.
	indexation:
		Array that associates a node to its position in the order (it is not necessary to count but will be to list)
	printing:
		If ``True`` will print triangles
		
	Returns
	-------
	nb_inter :
		Number of nodes in the intersection
	"""
	
	cdef int i, j, k
	cdef int v
	cdef long nb_inter = 0		# number of nodes in the intersection
	
	for k in range(indptr[u], indptr[u+1]):		# iterates over the neighbors of u
		v = indices[k]
		i = indptr[u]
		j = indptr[v]
		
		# calculates the intersection of the neighbors of u and v
		while (i < indptr[u+1]) and (j < indptr[v+1]):
			if indices[i] == indices[j]:
				# if (printing):
				#	printf("(%d,%d,%d)\n",indexation[u], indexation[v], indexation[i])
				
				i += 1
				j += 1
				nb_inter += 1	# increments the number of nodes in the intersection
			else :
				if indices[i] < indices[j]:
					i += 1
				else :
					j += 1
					
	return nb_inter
	
cdef long triangles_parallel_c(int n_nodes, int[:] indptr, int[:] indices, int[:] indexation):
	"""	   Counts the number of triangles in a DAG using a parallel range.

	Parameters
	----------
	n_nodes :
		Number of nodes.-
	indptr :
		CSR format index pointer array of the normalized adjacency matrix of a DAG.
	indices :
		CSR format index array of the normalized adjacency matrix of a DAG.
	indexation:
		Array that associates a node to its position in the order (it is not necessary to count but will be to list)
	printing:
		If ``True`` will print triangles
		
	Returns
	-------
	nb_triangles :
		Number of triangles in the graph
	"""
	cdef int u
	cdef long nb_triangles = 0		# number of triangles
	
	for u in prange(n_nodes, nogil=True):	# parallel range
		nb_triangles += tri_intersection(u, indptr, indices, indexation)
		
	return nb_triangles
	
cdef long fit_core(int n_nodes, int n_edges, int[:] indptr, int[:] indices, bint parallelize):
	"""	   Counts the number of triangles directly without exporting the graph.

	Parameters
	----------
	n_nodes :
		Number of nodes.
	n_edges :
		Number of edges
	indptr :
		CSR format index pointer array of the normalized adjacency matrix of a DAG.
	indices :
		CSR format index array of the normalized adjacency matrix of a DAG.
	parallelize :
		If ``True`` will use a parallel range to list triangles
	printing :
		If ``True`` will print triangles

	Returns
	-------
	nb_triangles :
		Number of triangles in the graph
	"""
	
	cdef int[:] ordered, indexation
	cdef int i
	cdef int u, v, k
	
	cdef np.ndarray[int, ndim=1] row
	cdef np.ndarray[int, ndim=1] column
	cdef np.ndarray[bool_type_t, ndim=1] data
	
	ordered = bucketSort(n_nodes, indptr)		# sorts the nodes in the graph
	indexation = np.empty((n_nodes,), dtype=np.int32)	# initializes an empty array
	for i in range(n_nodes):
		indexation[ordered[i]] = i
	
	# e = n_edges // 2		# new number of edges, half of the original number of edges for undirected graphs
	
	row = np.empty((n_edges,), dtype=np.int32)		# initializes an empty array
	column = np.empty((n_edges,), dtype=np.int32)		# once again initializes an array
	
	i = 0
	for u in range(n_nodes):
		for k in range(indptr[u], indptr[u+1]):
			v = indices[k]
			if indexation[u] < indexation[v]:	# the edge needs to be added
				row[i] = indexation[u]
				column[i] = indexation[v]
				i += 1
				
	row = row[:i]
	column = column[:i]
	data = np.ones((i,), dtype=bool)
	
	dag = sparse.csr_matrix((data, (row, column)), (n_nodes, n_nodes), dtype=bool)
	
	# counts/list the triangles in the DAG
	if (parallelize):
		return triangles_parallel_c(n_nodes, dag.indptr, dag.indices, ordered)	
	else:
		return triangles_c(n_nodes, dag.indptr, dag.indices, ordered)
		

class TriangleListing:
	""" Triangle listing algorithm which creates a DAG and list all triangles on it.

	* Graphs

	Parameters
	----------
	parallelize :
		If ``True``, uses a parallel range while listing the triangles.
	printing:
		If ``True```, will print each triangle found

	Attributes
	----------
	nb_tri : int
		Number of triangles

	Example
	-------
	>>> from sknetwork.topology import TriangleListing
	>>> from sknetwork.data import karate_club
	>>> tri = TriangleListing()
	>>> adjacency = karate_club()
	>>> tri.fit_transform(adjacency)
	45
	"""
	
	def __init__(self, parallelize : bool = False):
		# self.printing = printing
		self.parallelize = parallelize
		self.nb_tri = 0
		
	def fit(self, adjacency : sparse.csr_matrix) -> 'TriangleListing':
		""" Listing triangles.

		Parameters
		----------
		adjacency:
			Adjacency matrix of the graph.

		Returns
		-------
		 self: :class:`TriangleListing`
		"""
		
		self.nb_tri = fit_core(adjacency.shape[0], adjacency.nnz, adjacency.indptr,\
			adjacency.indices, self.parallelize)
		
		return self
		
	def fit_transform(self, *args, **kwargs) -> int:
		""" Fit algorithm to the data and return the number of triangles. Same parameters as the ``fit`` method.

		Returns
		-------
		nb_tri : int
			Number of triangles.
		"""
		self.fit(*args, **kwargs)
		return self.nb_tri
		
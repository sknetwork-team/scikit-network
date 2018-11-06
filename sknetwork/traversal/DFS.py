from model.graph import Graph
from utils.lazy_property import cached_property, DeleteCacheMixin
from traversal.algo import Algo

class DFS_algo(Algo):
	"""
	TODO use DeleteCacheMixin?
	"""

	def __init__(self, graph):
		super().__init__() 
		self._graph = graph
		self.parent = {}
		self.time = -1
		self.time_entry = {}
		self.time_exit = {}
		self.discovered = set()
		self.parent = {}
		self.processed = set()

	def iter(self, s, first = True,
		process_vertex_early=False, process_edge = False,
		process_vertex_late = False):
		"""
		DFS algorithm from source s
		inspired from http://www3.cs.stonybrook.edu/~skiena/algorist/book/

		Example
		-------
		>>>graph = { 0 : [1,4,5],
	          1 : [0,2,4],
	          2 : [1,3],
	          3 : [2,4],
	          4 : [0,1,3],
	          5 : [0]
	        }
		>>>G = Graph(graph)
		>>>DFS = DFS_algo(G)
		>>>for i in DFS.iter(1, process_vertex_early = True, process_edge = True):
			print(i)
		1
		(1, 0)
		0
		(0, 4)
		4
		(4, 1)
		(4, 3)
		3
		(3, 2)
		2
		(2, 1)
		(0, 5)
		5

		"""
		if first:
			self.time = -1
			self.time_entry = {}
			self.time_exit = {}
			self.discovered = set()
			self.parent = {}
			self.processed = set()

		self.discovered.add(s)
		self.time  += 1
		self.time_entry[s] = self.time
		if process_vertex_early:
			yield s
		for k in self._graph._graph_dict[s]:
			if k not in self.discovered:
				self.parent[k] = s
				if process_edge:
					yield s ,k
				yield from self.iter(k, first=False,
					process_vertex_early=process_vertex_early,
					process_edge=process_edge,
					process_vertex_late=process_vertex_late)
			elif (k not in self.processed and self.parent[s]!=k) or self._graph._directed:
                #corrected version see http://www3.cs.stonybrook.edu/~skiena/algorist/book/errata
				if process_edge:
					yield s,k
		if process_vertex_late:
			yield s
		self.time += 1
		self.processed.add(s)
		self.time_exit[s] = self.time


	def process_edge_cycle(self,u,v):
		if (v in self.discovered) and self.parent[u] != v:
			#corrected version see http://www3.cs.stonybrook.edu/~skiena/algorist/book/errata
			print('cycle form ', v, ' to ',u)
			self.find_path(v,u)
			print('\n')

	def finding_cycles(self,s):
		for (u,v) in self.iter(s, process_edge = True):
			self.process_edge_cycle(u,v)

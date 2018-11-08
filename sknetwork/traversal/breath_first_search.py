from model.graph import Graph
from collections import deque
from traversal.algo_traversal import AlgoTraversal


class BreathFirstSearchAlgo(AlgoTraversal):

    def __init__(self, graph):
        super().__init__()
        self._graph = graph
        self._discovered = {}

    def clear(self):
        self.parent = {}
        self._discovered = {}

    def iter(self, s, process_vertex_early=False, process_edge=False, process_vertex_late=False):
        """
		BFS algorithm from source s
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
		>>>g = Graph(graph)
		>>>bfs = BreathFirstSearchAlgo(g)
		>>>for i in bfs.iter(1, process_vertex_early = True, process_edge = True):
			print(i)
		1
		(1, 0)
		(1, 2)
		(1, 4)
		0
		(0, 4)
		(0, 5)
		2
		(2, 3)
		4
		(4, 3)
		5
		3
		"""
        q = deque([s])
        self._discovered = {s}
        processed = set()
        while q:
            vertex = q[0]
            if process_vertex_early:
                yield vertex
            processed.add(vertex)
            for k in self._graph.graph_dict[vertex]:
                if (k not in processed) or self._graph.directed:
                    if process_edge:
                        yield vertex, k
                if (k not in self._discovered):
                    q.append(k)
                    self._discovered.add(k)
            q.popleft()
            if process_vertex_late:
                yield vertex

    def tree(self, s):
        """
		constructs the BFS tree from source s

		"""
        self.parent = {i: -2 for i in self._graph.graph_dict.keys()}
        self.parent[s] = -1
        for (v, k) in self.iter(s, process_edge=True):
            if k not in self._discovered:
                self.parent[k] = int(v)

    def find_path_BFS(self, a, b):
        """
		find the shortest (unweighted) path from a to b
		"""
        self.tree(a)
        self.find_path(a, b)

    def connected_components(self):
        """
		print the connected components
		TODO find a nicer output and remove print
		:return: number of components, one seed per component
		"""
        self.clear()
        c = 0
        seeds = []
        for node in self._graph.graph_dict.keys():
            if node not in self._discovered:
                c += 1
                seeds.append(node)
                print('Component {}:'.format(c))
                for v in self.iter(node, process_vertex_early=True):
                    print('{}|'.format(v), end='')
                print()
        return c, seeds

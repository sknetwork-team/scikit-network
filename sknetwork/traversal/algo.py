from model.graph import Graph

class Algo(object):

	def __init__(self):
		self.parent = {}

	def find_path(self,a,b):
		"""
		find the path from a to b in the tree
		"""
		if self.parent == {}:
			raise ValueError("A tree must be computed first")

		if a == b:
			print('path:', a, end='')
		elif b == -1:
			print("There is no path from %d to " %a, end='')
		else:
			self.find_path(a,int(self.parent[b]))
			print(b, end='')
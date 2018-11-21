class AlgoTraversal(object):

    def __init__(self):
        self.parent = {}
        self.path = []

    def find_path(self, a, b, first = True, verbose = False):
        """
        find the path from a to b in the tree which needs to be computed first
        if no such path exists, return [b]
        """
        if first:
            self.path = []

        if self.parent == {}:
            raise ValueError("A tree must be computed first")

        if a == b:
            self.path.append(a)
            if verbose:
                print('path:', a, end='')
        elif b < 0:
            print("There is no path from %d to " %a, end='')
        else:
            self.find_path(a,int(self.parent[b]),first=False,verbose=verbose)
            self.path.append(b)
            if verbose:
                print(b, end='')
        return self.path

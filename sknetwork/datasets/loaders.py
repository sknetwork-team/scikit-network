from model.graph import Graph

def load_from_mtx(path, directed = True, comments="%", delimiter=None):
    """
    load a graph in mtx format see http://networkrepository.com/mtx-matrix-market-format.html

    Inputs
    ------
    path : str
        where the file is stored

    Returns
    -------
    a graph

    Example
    -------
    Download from http://nrvis.com/download/data/dimacs10/polblogs.zip
    and save in p = '/home/USER/scikit-network/data/polblogs.mtx'
    >>> G = load_from_mtx(p)
    loading graph with 1490 nodes and 19025 edges.
    number of nodes: 1224 (expected 1490); number of edges: 19025 (expected 19025)

    Note
    ----
    Useful repo : https://sparse.tamu.edu/SNAP

    """
    G = Graph()
    G.clear()
    G._directed = directed
    first_line = True
    with open(path, 'rt') as f:
        for line in f:
            p=line.find(comments)
            if p>=0:
                line = line[:p]
            if not len(line):
                continue
            # split line, should have 2 or more
            s=line.strip().split(delimiter)
            if len(s)<2:
                continue
            if first_line:
                try:                
                    n = s.pop(0)
                    m = s.pop(0)
                    e = s.pop(0)
                    if n != m:
                        print("check the number of nodes")
                    else:
                        print("loading graph with %s nodes and %s edges." %(n,e))
                        first_line = False
                except:
                    raise ValueError("First line should contain num_nodes num_nodes num_edges.")
                continue
            u=s.pop(0)
            v=s.pop(0)
            d=s
            G.add_edge(int(u),int(v),direct=True)
        
    G.basics()
    print("number of nodes: %d (expected %s); number of edges: %d (expected %s)" %(G.n_vertices,n,G.n_edges,e))
    return G

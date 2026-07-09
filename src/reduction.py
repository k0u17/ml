import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import squareform, pdist


def isomap(data, k, dim, returns_adjacency=False):
    n = data.shape[0]
    d = squareform(pdist(data, metric='sqeuclidean'))
    np.fill_diagonal(d, np.inf)
    graph = np.zeros_like(d)
    mask = (np.arange(n)[:, np.newaxis], np.argpartition(d, k, axis=1))
    graph[mask] = d[mask]
    graph = csr_matrix(graph)
    d_prime = dijkstra(csgraph=graph, directed=False)
    h = np.eye(n) - np.ones((n, n))/n
    g = -0.5 * h @ d_prime @ h
    _, v = linalg.eigh(g, subset_by_index=[n-dim, n-1])
    data @ v


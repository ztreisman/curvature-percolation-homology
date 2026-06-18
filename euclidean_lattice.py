"""Flat {3,6} triangular lattice patch: the Euclidean control case.

Standard triangular grid: vertex (i,j) for i in [0,n), j in [0,n).
Each vertex connects to (i+1,j), (i,j+1), (i+1,j-1) -- the three "forward"
neighbors of a triangular tiling -- which, taken together over the whole
patch, gives every vertex degree 6 in the interior (boundary vertices have
lower degree, same as the {3,7} disk has boundary effects at its outer ring).
"""
import networkx as nx


def triangular_lattice(n):
    """n x n patch of the {3,6} triangular lattice. Returns a networkx Graph."""
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            G.add_node((i, j))
    for i in range(n):
        for j in range(n):
            for di, dj in [(1, 0), (0, 1), (1, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    G.add_edge((i, j), (ni, nj))
    return G


if __name__ == "__main__":
    for n in [10, 15, 20, 25]:
        G = triangular_lattice(n)
        print(n, G.number_of_nodes(), G.number_of_edges(),
              "avg deg", 2 * G.number_of_edges() / G.number_of_nodes())

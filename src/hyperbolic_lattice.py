"""
Vertex labeling for {3,Q} hyperbolic tilings following Appendix B of
Mertens & Moore, arXiv:1708.05876.

Each vertex is represented as a tuple of ints (u_1,...,u_k) encoding the
sequence of edge labels on the canonical path from the origin.  This lets
us compute any vertex's full neighbor list on the fly, without storing the
entire lattice.

For P=3 (triangular faces, Q meeting at each vertex), two vertex types:
  e-vertex: 1 canonical incoming edge, Q-3 outgoing edges (labeled 0..Q-4)
  v-vertex: 2 incoming edges (canonical + right),  Q-4 outgoing edges (0..Q-5)
  origin:   Q outgoing edges (labeled 0..Q-1)

A vertex u=(u_1,...,u_k) is v-type iff its last label u_k equals 0 or
outdegree(parent(u))-1; otherwise e-type.  (Fig. 13 of Mertens & Moore.)
"""
import networkx as nx


def _outdegree(u, q):
    if len(u) == 0:
        return q
    if len(u) == 1:
        # All ring-1 vertices have exactly 1 incoming edge (from origin), so
        # they are e-type regardless of their label.  The v/e distinction only
        # applies from ring 2 onward, where sectors can share boundary vertices.
        return q - 3
    last = u[-1]
    pod = _outdegree(u[:-1], q)
    is_v = (last == 0 or last == pod - 1)
    return q - 4 if is_v else q - 3


def vertex_type(u, q):
    """'o' for origin, 'v' or 'e' for P=3 vertex."""
    if len(u) == 0:
        return 'o'
    if len(u) == 1:
        return 'e'  # ring-1: all e-type (1 incoming edge, from origin)
    pod = _outdegree(u[:-1], q)
    return 'v' if (u[-1] == 0 or u[-1] == pod - 1) else 'e'


def successor(u, q):
    """Counterclockwise ring neighbor; same layer (depth) as u."""
    if len(u) == 1:
        return ((u[0] + 1) % q,)
    parent, last = u[:-1], u[-1]
    pod = _outdegree(parent, q)
    if last < pod - 1:
        return child(parent, last + 1, q)
    return child(successor(parent, q), 0, q)


def predecessor(u, q):
    """Clockwise ring neighbor; same layer as u."""
    if len(u) == 1:
        return ((u[0] - 1) % q,)
    parent, last = u[:-1], u[-1]
    if last > 0:
        return parent + (last - 1,)
    w = predecessor(parent, q)
    x = _outdegree(w, q) - 1
    if vertex_type(u, q) == 'v':
        x -= 1
    return child(w, x, q)


def child(u, x, q):
    """Child of u reached via outgoing edge labeled x."""
    if len(u) == 0:
        return (x,)
    pod = _outdegree(u, q)
    if x < pod - 1:
        return u + (x,)
    # x == pod-1 leads to a v-vertex via its right incoming edge;
    # reroute to the canonical (left) path.
    return child(successor(u, q), 0, q)


def neighbors(u, q):
    """All Q neighbors of u in the {3,Q} tiling."""
    nbrs = []
    if len(u) > 0:
        nbrs.append(u[:-1])                   # canonical parent
        if vertex_type(u, q) == 'v':
            nbrs.append(predecessor(u[:-1], q))   # right parent
    nbrs.append(successor(u, q))
    nbrs.append(predecessor(u, q))
    od = _outdegree(u, q)
    for x in range(od):
        nbrs.append(child(u, x, q))
    return nbrs


def build_lattice(q, depth):
    """
    Build the {3,Q} tiling to BFS depth `depth` from the origin.
    Returns a networkx.Graph with tuple-labeled vertices.

    Memory: O(N) where N = total number of vertices.
    This avoids the exponential-size pre-generation needed for very deep
    ring-growth lattices (useful for q=12, 20 at large ring counts).
    """
    G = nx.Graph()
    G.add_node(())
    prev_layer = frozenset({()})

    for k in range(depth):
        next_layer = set()
        for u in prev_layer:
            od = _outdegree(u, q)
            for x in range(od):
                c = child(u, x, q)
                if c not in G:
                    G.add_node(c)
                    next_layer.add(c)
                G.add_edge(u, c)
        # same-layer ring edges for the newly created layer
        for v in next_layer:
            s = successor(v, q)
            if s in G:
                G.add_edge(v, s)
        prev_layer = frozenset(next_layer)

    return G


if __name__ == '__main__':
    import time
    from mesh_topology import generate_mesh_topology, to_graph

    for q in [7, 8, 12]:
        for depth in [3, 4, 5]:
            t0 = time.time()
            G_mm = build_lattice(q, depth)
            V_mm = G_mm.number_of_nodes()
            E_mm = G_mm.number_of_edges()

            # Compare against ring-growth construction
            pairs, tris, rc = generate_mesh_topology(rings=depth, q=q)
            G_rg = to_graph(pairs)
            V_rg = G_rg.number_of_nodes()
            E_rg = G_rg.number_of_edges()

            match = '✓' if (V_mm == V_rg and E_mm == E_rg) else '✗ MISMATCH'
            print(f"q={q} depth={depth}: V={V_mm}(MM) vs {V_rg}(rg)  "
                  f"E={E_mm} vs {E_rg}  {match}  [{time.time()-t0:.2f}s]")

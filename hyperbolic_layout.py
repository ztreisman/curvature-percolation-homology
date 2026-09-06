"""
Exact Poincare-disk placement for a {3,q} ring-growth graph, using the
graph's own triangular face list to derive the true rotational order of
edges around each vertex, and hyperbolic isometries (SU(1,1) matrices) to
place each vertex from an already-placed neighbor. No optimization: every
edge comes out at the exact target hyperbolic length by construction,
because {3,q} tiles H^2 exactly.
"""
import numpy as np


def edge_length(q):
    alpha = 2 * np.pi / q
    c = np.cos(alpha)
    return np.arccosh(c / (1 - c))


def rot_matrix(phi):
    return np.array([[np.exp(1j*phi/2), 0], [0, np.exp(-1j*phi/2)]], dtype=complex)


def trans_matrix(x):
    d = np.sqrt(1 - x*x)
    return np.array([[1/d, x/d], [x/d, 1/d]], dtype=complex)


def mobius_apply(M, z):
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    return (a*z + b) / (c*z + d)


def orient_faces_consistently(tris):
    """The raw tris list (transliterated from Rust code that never needed a
    global orientation) mixes CW- and CCW-listed triangles. Re-orient them
    consistently via the standard propagation rule for a triangulated
    orientable surface: BFS over the face-adjacency (dual) graph, forcing
    each face to traverse a shared edge in the direction opposite its
    already-oriented neighbor."""
    from collections import deque
    edge_faces = {}
    for fi, (a, b, c) in enumerate(tris):
        for u, v in [(a, b), (b, c), (c, a)]:
            edge_faces.setdefault(frozenset((u, v)), []).append(fi)
    for key, flist in edge_faces.items():
        assert len(flist) <= 2, f"non-manifold edge {key}: faces {flist}"

    oriented = [None] * len(tris)
    visited = [False] * len(tris)
    for start in range(len(tris)):
        if visited[start]:
            continue
        oriented[start] = tris[start]
        visited[start] = True
        dq = deque([start])
        while dq:
            fi = dq.popleft()
            a, b, c = oriented[fi]
            for u, v in [(a, b), (b, c), (c, a)]:
                for fj in edge_faces[frozenset((u, v))]:
                    if fj == fi or visited[fj]:
                        continue
                    third = [x for x in tris[fj] if x != u and x != v]
                    assert len(third) == 1
                    oriented[fj] = (v, u, third[0])
                    visited[fj] = True
                    dq.append(fj)
    assert all(o is not None for o in oriented)
    return oriented


def build_rotation_maps(tris):
    """From a consistently-oriented triangle list, build next[(v,u)] = w and
    prev[(v,w)] = u for every face (v,u,w) meaning 'at vertex v, the edge to
    w immediately follows the edge to u in rotational order' (and precedes
    it, for prev)."""
    tris = orient_faces_consistently(tris)
    nxt, prv = {}, {}
    for (v1, v2, v3) in tris:
        for a, b, c in [(v1, v2, v3), (v2, v3, v1), (v3, v1, v2)]:
            assert nxt.get((a, b), c) == c, f"inconsistent orientation at ({a},{b})"
            nxt[(a, b)] = c
            prv[(a, c)] = b
    return nxt, prv


def neighbor_ring_order(v, ref, nxt, prv, valid_neighbors):
    """Full cyclic (or partial, if v is on the outer boundary) order of v's
    neighbors starting at `ref`, walking forward via `nxt` and backward via
    `prv`, restricted to actual graph neighbors."""
    forward = []
    cur = ref
    seen = {ref}
    while True:
        nx_ = nxt.get((v, cur))
        if nx_ is None or nx_ not in valid_neighbors or nx_ in seen:
            break
        forward.append(nx_)
        seen.add(nx_)
        cur = nx_
    backward = []
    cur = ref
    while True:
        pv = prv.get((v, cur))
        if pv is None or pv not in valid_neighbors or pv in seen:
            break
        backward.append(pv)
        seen.add(pv)
        cur = pv
    return list(reversed(backward)), forward  # (before ref, after ref)


def place(G, tris, q, root=0):
    a = edge_length(q)
    v0 = np.tanh(a / 2)
    nxt, prv = build_rotation_maps(tris)

    M = {root: np.eye(2, dtype=complex)}
    pos = {root: 0.0 + 0.0j}
    ref_of = {root: None}
    order = [root]
    from collections import deque
    queue = deque([root])
    visited_processed = set()

    while queue:
        v = queue.popleft()
        if v in visited_processed:
            continue
        visited_processed.add(v)
        valid_neighbors = set(G.neighbors(v))

        if ref_of[v] is None:
            # root: no parent; pick an arbitrary neighbor as angle-0 reference
            ref = next(iter(valid_neighbors))
            before, after = neighbor_ring_order(v, ref, nxt, prv, valid_neighbors)
            seq = list(reversed(before)) + [ref] + after
            angles = {n: -k * 2*np.pi/q for k, n in enumerate(seq)}
        else:
            ref = ref_of[v]
            before, after = neighbor_ring_order(v, ref, nxt, prv, valid_neighbors)
            # 'before' = neighbors preceding ref in rotation order -> positive
            # angle offsets from pi; 'after' -> negative offsets from pi.
            angles = {ref: np.pi}
            for k, n in enumerate(after, start=1):
                angles[n] = np.pi - k * 2*np.pi/q
            for k, n in enumerate(before, start=1):
                angles[n] = np.pi + k * 2*np.pi/q

        for n, phi in angles.items():
            if n not in pos:
                Mv = M[v]
                pos[n] = mobius_apply(Mv, v0 * np.exp(1j*phi))
                M[n] = Mv @ rot_matrix(phi) @ trans_matrix(v0)
                ref_of[n] = v
                order.append(n)
            if n not in visited_processed:
                queue.append(n)

    return pos


def validate(G, pos, target, tol=1e-6):
    errs = []
    for u, v in G.edges():
        z, w = pos[u], pos[v]
        num = 2*abs(z-w)**2
        den = (1-abs(z)**2)*(1-abs(w)**2)
        d = np.arccosh(max(1.0, 1+num/den))
        errs.append(abs(d-target))
    errs = np.array(errs)
    print(f"edge length error: max={errs.max():.3e} mean={errs.mean():.3e} "
          f"(target a={target:.6f}), {np.sum(errs>tol)}/{len(errs)} exceed tol={tol}")
    return errs


if __name__ == "__main__":
    from mesh_topology import generate_mesh_topology, to_graph
    for rings in [1, 2, 3, 4, 5]:
        pairs, tris, rc = generate_mesh_topology(rings=rings, q=7)
        G = to_graph(pairs)
        a = edge_length(7)
        pos = place(G, tris, 7)
        print(f"rings={rings} N={G.number_of_nodes()} E={G.number_of_edges()} placed={len(pos)}")
        validate(G, pos, a)

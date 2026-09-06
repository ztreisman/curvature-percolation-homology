"""
Figure: the {3,7} tiling in the Poincare disk, bond-percolated at p_c, with
one H1 cycle of the occupied subgraph highlighted.

Pipeline:
  1. Build the {3,7} combinatorial graph (mesh_topology, ring-growth, already
     used and validated throughout the rest of this project).
  2. Lay it out in the Poincare disk model of H^2 by (a) an informed initial
     guess that respects the graph's known ring/angular structure, then
     (b) gradient-descent relaxation on hyperbolic edge length so every edge
     has EXACTLY the {3,7} tiling's true edge length
         a = arccosh( cos(2*pi/7) / (1 - cos(2*pi/7)) )
     (the standard hyperbolic law-of-cosines formula for an equilateral
     triangle with vertex angle 2*pi/7 -- 7 such triangles tile 2*pi exactly
     around each vertex). This is an exact isometric embedding into H^2
     itself (the tiling's native space), not the finite-dimensional
     Euclidean embedding-capacity problem studied elsewhere in this project;
     {3,7} tiles H^2 perfectly by construction, so this always succeeds.
  3. Run one bond-percolation realization at p = p_c({3,7}) = 0.199351
     (Mertens & Moore, arXiv:1708.05876, Table I).
  4. Find a cycle in the occupied subgraph (a graph-theoretic H1 generator
     of the 1-skeleton, i.e. of the occupied edge set as a graph -- distinct
     from the flag-complex construction used elsewhere in the paper, which
     fills in triangles and would make a single triangle loop's persistence
     zero; here we want a visible, uncapped loop).
  5. Draw the whole tiling faintly, the occupied subgraph solidly, and the
     chosen cycle in a bold highlight color, all as true hyperbolic geodesic
     arcs (circular arcs orthogonal to the unit circle), not straight lines.
"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import json

from mesh_topology import generate_mesh_topology, to_graph, ring_counts_seq
from hyperbolic_layout import place as exact_place, edge_length

Q = 7
RINGS = 4
P_C = 0.199351  # Mertens & Moore Table I, {3,7} bond percolation threshold
SEED_SEARCH = range(200)  # percolation seeds to search for a draw with beta_1 >= 1


# ---------- 1. combinatorics + exact hyperbolic layout ----------
# (layout is delegated entirely to hyperbolic_layout.place, which derives a
# consistent rotation system from the triangulation's face list and places
# every vertex via exact hyperbolic isometries -- see that module for why
# the earlier gradient-descent approach got stuck in a local minimum.)

def build_graph():
    pairs, tris, rc = generate_mesh_topology(rings=RINGS, q=Q)
    G = to_graph(pairs)
    return G, tris


# ---------- 3. percolation ----------

def percolate(G, p, seed):
    rng = np.random.default_rng(seed)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if rng.random() < p:
            H.add_edge(u, v)
    return H


def filled_triangles(H, tris):
    """Triangles of the original tiling whose three edges are all occupied
    -- these get a 2-simplex filled in by the same construction used for
    the paper's own persistent-homology pipeline (percolation_topology.py),
    so their boundary is trivial (a boundary, not a cycle) in H_1 of the
    flag complex, even though it is a perfectly good 3-cycle of the bare
    occupied graph."""
    out = []
    for (a, b, c) in tris:
        if H.has_edge(a, b) and H.has_edge(b, c) and H.has_edge(c, a):
            out.append((a, b, c))
    return out


def h1_basis_of_flag_complex(H, tris):
    """A basis for H_1 of the flag complex (occupied edges + filled
    triangles as 2-cells), as edge sets -- i.e. the graph's cycle space Z_1
    modulo the subspace B_1 spanned by filled-triangle boundaries, computed
    by Gaussian elimination over GF(2) (edge sets as bitmasks). A bare
    occupied triangle contributes a boundary, so it reduces to zero and is
    correctly excluded, unlike a plain graph cycle basis."""
    edges = list(H.edges())
    edge_id = {frozenset(e): i for i, e in enumerate(edges)}

    def vec_of(edge_list):
        v = 0
        for e in edge_list:
            v ^= 1 << edge_id[frozenset(e)]
        return v

    pivot_of = {}

    def reduce_vec(v):
        while v:
            hi = v.bit_length() - 1
            if hi in pivot_of:
                v ^= pivot_of[hi]
            else:
                break
        return v

    for (a, b, c) in filled_triangles(H, tris):
        v = reduce_vec(vec_of([(a, b), (b, c), (c, a)]))
        if v:
            pivot_of[v.bit_length() - 1] = v

    accepted = []
    for comp in nx.connected_components(H):
        sub = H.subgraph(comp)
        if sub.number_of_edges() < sub.number_of_nodes():
            continue  # tree, contributes nothing to Z_1
        try:
            basis = nx.minimum_cycle_basis(sub)
        except Exception:
            continue
        for cyc in basis:
            if len(cyc) < 3:
                continue
            v = reduce_vec(vec_of(cycle_edges(H, cyc)))
            if v:
                pivot_of[v.bit_length() - 1] = v
                accepted.append(v)

    return [[edges[i] for i in range(len(edges)) if v & (1 << i)] for v in accepted]


def cycle_edges(H, cycle_nodes):
    """Reconstruct an edge cycle (ordered) from a node set returned by
    minimum_cycle_basis (which gives nodes, not an ordered cycle)."""
    sub = H.subgraph(cycle_nodes).copy()
    # a genuine simple cycle on these nodes should be 2-regular
    if all(d == 2 for _, d in sub.degree()):
        order = list(nx.cycle_basis(sub)[0]) if nx.cycle_basis(sub) else list(sub.nodes())
        # find an actual Eulerian-ish ordering by walking the 2-regular graph
        start = order[0]
        path = [start]
        prev = None
        cur = start
        while True:
            nbrs = [n for n in sub.neighbors(cur) if n != prev]
            nxt = nbrs[0]
            if nxt == start:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        return list(zip(path, path[1:] + [path[0]]))
    return list(sub.edges())


# ---------- 4. geodesic drawing ----------

def circumcenter(p1, p2, p3):
    ax, ay = p1.real, p1.imag
    bx, by = p2.real, p2.imag
    cx, cy = p3.real, p3.imag
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / d
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / d
    return complex(ux, uy)


def geodesic_points(z1, z2, n=60):
    if abs(z1) < 1e-6 or abs(z2) < 1e-6:
        return np.linspace(z1, z2, n)
    cross = z1.real * z2.imag - z1.imag * z2.real
    if abs(cross) < 1e-7:
        return np.linspace(z1, z2, n)
    z1_inv = 1 / np.conj(z1)
    c = circumcenter(z1, z2, z1_inv)
    r = abs(z1 - c)
    a1 = np.angle(z1 - c)
    a2 = np.angle(z2 - c)
    for cand in (a2 - a1, a2 - a1 + 2*np.pi, a2 - a1 - 2*np.pi):
        angs = a1 + np.linspace(0, cand, n)
        pts = c + r * np.exp(1j * angs)
        if np.all(np.abs(pts) <= 1.0 + 1e-6):
            return pts
    return np.linspace(z1, z2, n)  # fallback


# ---------- main ----------

def make_figure(G, tris, pos, p, label, out_path, seed_search=SEED_SEARCH):
    basis, seed_used = None, None
    for seed in seed_search:
        H = percolate(G, p, seed)
        b = h1_basis_of_flag_complex(H, tris)
        if b:
            basis, seed_used = b, seed
            break
    if basis is None:
        raise RuntimeError(f"no nontrivial H_1 class found in any searched "
                            f"percolation draw at p={p}")

    H = percolate(G, p, seed_used)
    highlight_edges = set()
    highlight_nodes = set()
    for edge_list in basis:
        for u, v in edge_list:
            highlight_edges.add(frozenset((u, v)))
            highlight_nodes.update((u, v))
    n_filled = len(filled_triangles(H, tris))
    graph_beta1 = H.number_of_edges() - H.number_of_nodes() + nx.number_connected_components(H)
    beta1 = len(basis)  # true H_1 rank of the flag complex (graph_beta1 minus filled-triangle relations)
    print(f"p={p:.4f} ({label})  seed={seed_used}  occupied edges={H.number_of_edges()}  "
          f"filled triangles={n_filled}  graph beta_1={graph_beta1}  "
          f"flag-complex beta_1={beta1}  highlighted edges={len(highlight_edges)}")

    fig, ax = plt.subplots(figsize=(9, 9))
    boundary = plt.Circle((0, 0), 1.0, fill=False, color="black", lw=1.2, zorder=1)
    ax.add_patch(boundary)

    for u, v in G.edges():
        pts = geodesic_points(pos[u], pos[v])
        ax.plot(pts.real, pts.imag, color="0.85", lw=0.6, zorder=2)

    for a, b, c in filled_triangles(H, tris):
        boundary_pts = np.concatenate([
            geodesic_points(pos[a], pos[b]),
            geodesic_points(pos[b], pos[c]),
            geodesic_points(pos[c], pos[a]),
        ])
        tri = plt.Polygon(
            np.column_stack([boundary_pts.real, boundary_pts.imag]),
            closed=True, facecolor="#f4b6b6", edgecolor="none", alpha=0.7, zorder=2.5,
        )
        ax.add_patch(tri)

    for u, v in H.edges():
        if frozenset((u, v)) in highlight_edges:
            continue
        pts = geodesic_points(pos[u], pos[v])
        ax.plot(pts.real, pts.imag, color="black", lw=1.4, zorder=3)

    for uv in highlight_edges:
        u, v = tuple(uv)
        pts = geodesic_points(pos[u], pos[v])
        ax.plot(pts.real, pts.imag, color="#d62728", lw=3.2, zorder=4)

    occ_nodes = [n for n in G.nodes() if H.degree(n) > 0 and n not in highlight_nodes]
    xs = [pos[n].real for n in occ_nodes]
    ys = [pos[n].imag for n in occ_nodes]
    ax.scatter(xs, ys, s=6, color="black", zorder=3)
    cxs = [pos[n].real for n in highlight_nodes]
    cys = [pos[n].imag for n in highlight_nodes]
    ax.scatter(cxs, cys, s=28, color="#d62728", zorder=5)

    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"$\\{{3,{Q}\\}}$ tiling, bond percolation at $p={p:.4f}$ ({label}, "
        f"$p_c={P_C}$)\n"
        f"gray: full tiling   pink: filled (capped) triangles   black: other "
        f"occupied edges\nred: all $\\beta_1={beta1}$ independent $H_1$ classes "
        f"of the flag complex ({n_filled} filled triangles excluded)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"saved {out_path}")

    return {
        "p": p, "label": label, "seed": seed_used, "beta_1": beta1,
        "n_basis_cycles": len(basis), "n_highlighted_edges": len(highlight_edges),
        "n_occupied_edges": H.number_of_edges(),
    }


def main():
    G, tris = build_graph()
    print(f"{{3,{Q}}} ring {RINGS}: N={G.number_of_nodes()} E={G.number_of_edges()}")

    a = edge_length(Q)
    print(f"target hyperbolic edge length a = {a:.6f}")

    pos = exact_place(G, tris, Q)
    errs = [abs(np.arccosh(max(1.0, 1 + 2*abs(pos[u]-pos[v])**2 /
            ((1-abs(pos[u])**2)*(1-abs(pos[v])**2)))) - a) for u, v in G.edges()]
    print(f"exact placement: max edge-length error = {max(errs):.2e}")

    runs = [
        (P_C,       "critical",     "figure_poincare_percolation.png"),
        (0.5*P_C,   "subcritical",  "figure_poincare_percolation_subcritical.png"),
        (1.5*P_C,   "supercritical","figure_poincare_percolation_supercritical1.png"),
        (2.5*P_C,   "supercritical","figure_poincare_percolation_supercritical2.png"),
    ]
    meta = []
    for p, label, out_path in runs:
        meta.append(make_figure(G, tris, pos, p, label, out_path))

    with open("poincare_percolation_meta.json", "w") as f:
        json.dump({
            "q": Q, "rings": RINGS, "N": G.number_of_nodes(), "E": G.number_of_edges(),
            "p_c": P_C, "edge_length_a": a, "runs": meta,
        }, f, indent=2)


if __name__ == "__main__":
    main()

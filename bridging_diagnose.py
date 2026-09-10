"""
Isolate the 20% shortfall in the analytic bridging term.

(1) Abstract 1D Monte Carlo of the overcount model: a long cycle, rim edges
    Bernoulli(p), inward spoke per vertex Bernoulli(p), and each absent rim
    edge (seam) bridged independently with prob beta. Measure J - N directly
    and compare with the closed form c(p) - c_eff(p). If they agree, the
    abstract accounting is right and the error is in beta.

(2) Measure beta(p) directly on the real graph: for each seam (absent rim
    edge) between u, u+1 in ring k, is u connected to u+1 using only ring
    k+1 vertices and their spokes to u, u+1? Compare with beta = (p/(1-p+p^2))^2.
"""
import numpy as np
import networkx as nx
from mesh_topology import generate_mesh_topology, to_graph


def beta_analytic(p):
    return (p / (1 - p + p * p)) ** 2


def c_wheel(p):
    return (1 - p) ** 3 / (1 - p + p * p)


def c_eff(p, beta):
    pe = p + (1 - p) * beta
    return (1 - pe) ** 2 * (1 - p) / (1 - pe * (1 - p))


def abstract_overcount(p, beta, m=200000, rng=None):
    """J - N per vertex for the abstract seam-bridging model on a cycle of m vertices."""
    rim = rng.random(m) < p            # rim edge i connects vertex i and i+1
    spoke = rng.random(m) < p          # inward spoke present at vertex i
    bridge = rng.random(m) < beta      # seam bridge available at rim edge i (if absent)
    # original arcs: maximal runs under rim; super-arcs: under rim | (~rim & bridge)
    eff = rim | (~rim & bridge)

    def count_sf_runs(conn):
        # number of maximal runs (cyclic) with no present spoke
        # break the cycle at an absent connection to linearize
        idx = np.where(~conn)[0]
        if len(idx) == 0:
            return 0 if spoke.any() else 1
        start = (idx[0] + 1) % m
        order = (np.arange(m) + start) % m
        conn_o = conn[order]
        spoke_o = spoke[order]
        n_sf = 0
        cur_has_spoke = False
        for i in range(m):
            cur_has_spoke |= spoke_o[i]
            if not conn_o[i]:  # run ends after vertex i
                if not cur_has_spoke:
                    n_sf += 1
                cur_has_spoke = False
        return n_sf

    J = count_sf_runs(rim)
    N = count_sf_runs(eff)
    return (J - N) / m


rng = np.random.default_rng(0)
print("=== (1) abstract model: MC (J-N)/m vs c - c_eff ===")
for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    b = beta_analytic(p)
    mc = np.mean([abstract_overcount(p, b, rng=rng) for _ in range(3)])
    print(f"  p={p:.1f}  beta={b:.4f}  MC={mc:.5f}  formula={c_wheel(p)-c_eff(p,b):.5f}")


print("\n=== (2) direct beta(p) on the real graph (q=20, rings=3, seam in ring 2 bridged via ring 3) ===")
q, rings = 20, 3
pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
G = to_graph(pairs)
ring = nx.single_source_shortest_path_length(G, 0)
k = rings - 1
ringk = [v for v in G.nodes() if ring[v] == k]
outer = [v for v in G.nodes() if ring[v] == k + 1]
rim_k = [(u, v) for u, v in G.edges() if ring[u] == k and ring[v] == k]
edges = list(G.edges())
rng = np.random.default_rng(2)
for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    n_seam, n_bridged = 0, 0
    for _ in range(20):
        w = rng.random(len(edges))
        pres = set(frozenset(e) for e, x in zip(edges, w) if x <= p)
        # graph on ring k+1 vertices plus ring-k vertices, using only present
        # rim edges within ring k+1 and present spokes between k and k+1
        H = nx.Graph()
        H.add_nodes_from(ringk)
        H.add_nodes_from(outer)
        for u, v in edges:
            if frozenset((u, v)) not in pres:
                continue
            ru, rv = ring[u], ring[v]
            if (ru == k + 1 and rv == k + 1) or {ru, rv} == {k, k + 1}:
                H.add_edge(u, v)
        comp = {}
        for i, c in enumerate(nx.connected_components(H)):
            for v in c:
                comp[v] = i
        for u, v in rim_k:
            if frozenset((u, v)) in pres:
                continue  # not a seam
            n_seam += 1
            if comp[u] == comp[v]:
                n_bridged += 1
    print(f"  p={p:.1f}  measured beta={n_bridged/n_seam:.4f}   analytic beta={beta_analytic(p):.4f}   "
          f"ratio={n_bridged/n_seam/beta_analytic(p):.3f}   (seams={n_seam})")

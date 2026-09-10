"""
Test the first-order analytic bridging term against simulation.

At a rim break (u,u+1) in ring k, the shared v-type child w in ring k+1
has spokes to both u and u+1. The rim-connected run through w bridges the
seam iff it carries a present spoke into each side. In the large-lambda
limit (long own-children blocks) this gives:

  seam bridge prob   beta(p)  = [p / (1-p+p^2)]^2
  effective rim conn p_eff(p) = p + (1-p) beta = p / (1-p+p^2)
  per-inner-vertex overcount
     b(p) = c(p) - c_eff(p) = p^2 (1-p)^3 / [(1-p+p^2)(1-2p+2p^2)]

where c(p) = (1-p)^3/(1-p+p^2) is the wheel density at alpha -> 0 and
c_eff uses p_eff for rim connectivity but p for spoke presence.

Prediction: residual(q) = B_1 / lambda(q) with B_1 = int_0^1 b(p) dp.
"""
import numpy as np
import networkx as nx
from scipy.integrate import quad
from mesh_topology import generate_mesh_topology, to_graph

MEASURED = {
    7: 0.24557, 8: 0.25852, 9: 0.26548, 10: 0.26990, 11: 0.27297,
    12: 0.27532, 13: 0.27700, 14: 0.27837, 16: 0.28055, 18: 0.28198,
    20: 0.28310,
}


def lam(q):
    a = q - 4
    return (a + np.sqrt(a * a - 4)) / 2


def c_wheel(p, alpha):
    return (1 - p) ** 3 * (1 - alpha * p) / (1 - p * (1 - p) * (1 - alpha * p))


def mu_wheel(q):
    return quad(lambda p: c_wheel(p, 1 / lam(q)), 0, 1)[0]


def b_analytic(p):
    return p ** 2 * (1 - p) ** 3 / ((1 - p + p * p) * (1 - 2 * p + 2 * p * p))


B1 = quad(b_analytic, 0, 1)[0]
print(f"analytic B_1 = int b(p) dp = {B1:.5f}")

print("\n=== residual * lambda vs B_1 ===")
Bs = []
for q, m in MEASURED.items():
    r = mu_wheel(q) - m
    Bs.append(r * lam(q))
    print(f"q={q:3d}  residual*lambda={r*lam(q):.5f}   "
          f"mu_wheel - B1/lambda = {mu_wheel(q) - B1/lam(q):.5f}  (measured {m:.5f}, "
          f"diff {mu_wheel(q) - B1/lam(q) - m:+.5f})")
Bs = np.array(Bs)
print(f"empirical B: mean {Bs.mean():.5f}, std {Bs.std():.5f}")


# ---------- direct simulation of the inner-ring overcount ----------

def build(q, rings):
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
    G = to_graph(pairs)
    ring = nx.single_source_shortest_path_length(G, 0)
    rim = [(u, v) for u, v in G.edges() if ring[u] == ring[v]]
    spoke = [(u, v) for u, v in G.edges() if ring[u] != ring[v]]
    return G, ring, rim, spoke


def decompose(G, ring, rim, spoke, p, rng, nrings):
    edges = list(G.edges())
    w = rng.random(len(edges))
    pres = set(frozenset(e) for e, x in zip(edges, w) if x <= p)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(e for e in edges if frozenset(e) in pres)

    Rim = nx.Graph()
    Rim.add_nodes_from(G.nodes())
    Rim.add_edges_from(e for e in rim if frozenset(e) in pres)
    arc_id, arc_ring = {}, {}
    for i, comp in enumerate(nx.connected_components(Rim)):
        for v in comp:
            arc_id[v] = i
        arc_ring[i] = ring[next(iter(comp))]
    has_inward = set()
    for u, v in spoke:
        if frozenset((u, v)) in pres:
            outer = u if ring[u] > ring[v] else v
            has_inward.add(arc_id[outer])
    J = np.zeros(nrings + 1)
    for a, k in arc_ring.items():
        if k > 0 and a not in has_inward:
            J[k] += 1
    N = np.zeros(nrings + 1)
    for comp in nx.connected_components(H):
        if 0 in comp:
            continue
        N[min(ring[v] for v in comp)] += 1
    return J, N


for q, rings, ntrial in [(20, 3, 30), (12, 4, 12)]:
    G, ring, rim, spoke = build(q, rings)
    V = G.number_of_nodes()
    rc = np.bincount([ring[v] for v in G.nodes()])
    # only the ring just inside the outermost is bridged by exactly one ring
    k = rings - 1
    print(f"\n=== q={q}, rings={rings}, N={V}, lambda={lam(q):.2f}; "
          f"testing ring {k} (rc={rc[k]}) bridged only by ring {rings} ===")
    rng = np.random.default_rng(1)
    ps = np.linspace(0.05, 0.95, 19)
    prof = []
    for p in ps:
        Jsum = np.zeros(rings + 1)
        Nsum = np.zeros(rings + 1)
        for _ in range(ntrial):
            J, N = decompose(G, ring, rim, spoke, p, rng, rings)
            Jsum += J
            Nsum += N
        over = (Jsum[k] - Nsum[k]) / ntrial / rc[k]
        prof.append(over)
        print(f"  p={p:.2f}  sim b(p)={over:.4f}   analytic b(p)={b_analytic(p):.4f}   "
              f"ratio={over/b_analytic(p) if b_analytic(p)>0 else float('nan'):.3f}")
    print(f"  sim integral ~ {np.trapezoid(prof, ps):.4f}  vs analytic B_1 = {B1:.4f}")

"""
Structure of the e-type / v-type sequence around a ring.

Claims tested:
  (0) ring n is sigma(ring n-1) up to rotation, where sigma is the
      substitution e -> V e^(q-5), V -> V e^(q-6); so ring n = sigma^(n-1)(e^q)
  (1) v_n = rc[n-1] exactly (each adjacent pair in ring n-1 shares one child)
  (2) the type sequence around ring n is periodic with period rc[n]/q,
      the fundamental domain of the tiling's q-fold rotational symmetry
  (3) within a period it is balanced: every window of length L holds
      floor(L*a) or ceil(L*a) v-types, a = rc[n-1]/rc[n]
"""
import os
from _paths import HERE, RESULTS, FIGURES
import networkx as nx
from mesh_topology import generate_mesh_topology, to_graph, ring_counts_seq


def type_sequence(q, rings):
    pairs, tris, rc = generate_mesh_topology(rings=rings, q=q)
    G = to_graph(pairs)
    depth = nx.single_source_shortest_path_length(G, 0)
    last = [v for v in G.nodes() if depth[v] == rings]
    prev = set(v for v in G.nodes() if depth[v] == rings - 1)
    rim = nx.Graph()
    rim.add_nodes_from(last)
    rim.add_edges_from((u, v) for u, v in G.edges()
                       if depth.get(u) == rings and depth.get(v) == rings)
    start = min(last)
    order, prv, cur = [start], None, start
    while True:
        nxt = [n for n in rim.neighbors(cur) if n != prv]
        if not nxt or nxt[0] == start:
            break
        order.append(nxt[0])
        prv, cur = cur, nxt[0]
    return "".join("V" if sum(1 for u in G.neighbors(v) if u in prev) == 2 else "e"
                   for v in order)


def minimal_cyclic_period(s):
    n = len(s)
    for p in range(1, n + 1):
        if n % p == 0 and s == s[:p] * (n // p):
            return p
    return n


def is_balanced(s, a):
    n = len(s)
    d = s + s  # cyclic windows
    for L in range(1, min(n, 200) + 1):
        counts = {d[i:i + L].count("V") for i in range(n)}
        lo = int(L * a)
        if not counts <= {lo, lo + 1}:
            return False, L
    return True, None


def substitute(s, q):
    """Ring n from ring n-1: e -> V e^(q-5), V -> V e^(q-6)."""
    return "".join("V" + "e" * (q - 5) if c == "e" else "V" + "e" * (q - 6) for c in s)


def equal_up_to_rotation(a, b):
    return len(a) == len(b) and (a in b + b)


print("=== substitution rule: ring n = sigma(ring n-1), up to rotation ===")
for q, depths in [(7, [3, 4, 5, 6]), (8, [3, 4, 5]), (9, [3, 4]), (12, [3, 4]), (20, [3])]:
    for n in depths:
        prev_s, cur_s = type_sequence(q, n - 1), type_sequence(q, n)
        ok = equal_up_to_rotation(substitute(prev_s, q), cur_s)
        ok_rev = equal_up_to_rotation(substitute(prev_s[::-1], q), cur_s)
        print(f"  q={q:2d} ring {n-1}->{n}: len {len(prev_s)}->{len(cur_s)}  "
              f"match={'yes' if ok else ('yes (reversed orientation)' if ok_rev else 'NO')}")

for q, depths in [(7, [4, 5, 6]), (8, [3, 4, 5]), (12, [3, 4]), (20, [2, 3])]:
    rc = ring_counts_seq(q, max(depths) + 1)
    print(f"\nq = {q}   rc = {rc[:max(depths)+1]}")
    for n in depths:
        s = type_sequence(q, n)
        v = s.count("V")
        per = minimal_cyclic_period(s)
        a = rc[n - 1] / rc[n] if n >= 2 else 0.0
        bal, badL = is_balanced(s, a)
        print(f"  ring {n}: len={len(s):5d}  v={v:5d} (rc[n-1]={rc[n-1]:5d} "
              f"{'ok' if v == rc[n-1] else 'MISMATCH'})   "
              f"period={per:5d} (rc[n]/q={rc[n]//q:5d} "
              f"{'ok' if per == rc[n] // q else 'differs'})   "
              f"slope={a:.6f}  balanced={'yes' if bal else f'no at L={badL}'}")

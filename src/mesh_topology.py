"""
Direct port of the combinatorial generation logic in `Mesh::new()` from
main.rs (the ring_counts / pairs / tris construction only -- NOT the
physics relaxer, which only moves point coordinates and never touches
`pairs`/`tris` after construction, confirmed by reading do_iteration()).

This lets us get the exact {3,q} tiling graph and face list that the
existing pipeline already grows, with no dependency on Rust or on running
the relaxer.
"""
import networkx as nx


def ring_counts_seq(q, rings):
    rc = [0, q]
    for i in range(2, rings + 1):
        rc.append((q - 4) * rc[i - 1] - rc[i - 2])
    rc = [1 if x == 0 else x for x in rc]
    return rc


def generate_mesh_topology(rings, q=7):
    """Literal transliteration of main.rs's pairs/tris construction.
    Returns (pairs, tris, ring_counts) using the SAME index arithmetic as
    the Rust code (offset = sum(ring_counts[:ring])), bugs and all, so we
    can directly inspect what the existing pipeline actually produces."""
    rc = ring_counts_seq(q, rings)

    pairs = []
    tris = []
    n_points = 1 + sum(rc[1:rings + 1])
    cusps = [False] * (n_points + 5)  # pad defensively

    # Manually prepare first ring
    for i in range(1, q + 1):
        a = i
        b = i % q + 1
        pairs.append((a, b))
        pairs.append((0, i))
        tris.append((0, i, b))

    for ring in range(2, rings + 1):
        offset = sum(rc[:ring])
        prev_offset = sum(rc[:ring - 1])
        cur_previous = offset - 1
        to_next_cusp = 0
        for i in range(rc[ring]):
            index = i + offset
            if to_next_cusp == 0:
                cusps[index] = True
                pairs.append((index, cur_previous))
                temp = (cur_previous - prev_offset + 1) % rc[ring - 1] + prev_offset
                pairs.append((index, temp))
                tris.append((index, cur_previous, temp))
                cur_previous = temp
                to_next_cusp = q - (6 if cusps[cur_previous] else 5)
            else:
                pairs.append((index, cur_previous))
                to_next_cusp -= 1
            next_index = (index - offset + 1) % rc[ring] + offset
            pairs.append((index, next_index))
            tris.append((index, next_index, cur_previous))

    return pairs, tris, rc


def to_graph(pairs):
    G = nx.Graph()
    G.add_edges_from(pairs)
    return G


if __name__ == "__main__":
    pairs, tris, rc = generate_mesh_topology(rings=5, q=7)
    G = to_graph(pairs)
    print("ring_counts (rc[0..5]):", rc[:6])
    print("cumulative through ring5 (1 + sum rc[1:6]):", 1 + sum(rc[1:6]))
    print("n_vertices in graph:", G.number_of_nodes())
    print("n_edges:", G.number_of_edges())
    print("n_tris:", len(tris))
    print("max label used:", max(max(p) for p in pairs))

    from collections import Counter
    deg_counts = Counter(dict(G.degree()).values())
    print("degree distribution:", sorted(deg_counts.items()))

    # BFS ring structure from vertex 0
    bfs_dist = nx.single_source_shortest_path_length(G, 0)
    ring_sizes = Counter(bfs_dist.values())
    print("BFS ring sizes from vertex 0:", sorted(ring_sizes.items()))

"""
Larger critical-percolation figure at ring 8 (N~11,000), to check whether
the sparse look of the ring-4 critical figure is a small-sample artifact.
Reuses figure_poincare_percolation.py's pipeline at a bumped ring depth,
critical draw only (p_c), since pc_ring_scan.py already confirmed the H1
count itself is not the interesting new thing here -- the picture is.
"""
import os
from _paths import HERE, RESULTS, FIGURES
import time
import figure_poincare_percolation as fpp

fpp.RINGS = 8

t0 = time.time()
G, tris = fpp.build_graph()
print(f"{{3,{fpp.Q}}} ring {fpp.RINGS}: N={G.number_of_nodes()} E={G.number_of_edges()}  "
      f"[{time.time()-t0:.1f}s]")

a = fpp.edge_length(fpp.Q)
pos = fpp.exact_place(G, tris, fpp.Q)
print(f"placed  [{time.time()-t0:.1f}s]")

fpp.make_figure(G, tris, pos, fpp.P_C, "critical",
                os.path.join(FIGURES, "figure_poincare_percolation_ring8_critical.png"))
print(f"done  [{time.time()-t0:.1f}s]")

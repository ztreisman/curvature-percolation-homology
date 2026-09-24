"""
Rebuild paper/arxiv/, the flattened arXiv submission: the tex with
\\graphicspath removed, the .bbl, and the figures it uses, all in one
directory. Compile the main paper (pdflatex, bibtex, pdflatex x2) first so
the .bbl is current.

Suggested categories: math.PR (primary); cross-list math.AT, cond-mat.stat-mech.
"""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "arxiv")
NAME = "curvature-percolation-homology"

tex = open(os.path.join(HERE, NAME + ".tex"), encoding="utf8").read()
tex = tex.replace("\\graphicspath{{../figures/}}\n", "")
figs = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)))

os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))
open(os.path.join(OUT, NAME + ".tex"), "w", encoding="utf8", newline="\n").write(tex)
shutil.copy(os.path.join(HERE, NAME + ".bbl"), OUT)
for f in figs:
    shutil.copy(os.path.join(HERE, "..", "figures", f), OUT)
print("wrote", OUT, "with", len(figs), "figures")

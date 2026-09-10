"""
Check a candidate closed-form derivation of mu(q) := lim E[w_MST]/V against
the measured table in the paper.

Derivation sketch: build the MST ring by ring. Once ring n-1 is fully
connected, the weight added by ring n equals the MST weight of ring n's own
rim cycle plus its spoke edges to a single contracted "interior" vertex S
(standard MST contraction fact). Each ring-n vertex is e-type (1 spoke to S)
or v-type (2 independent spokes to S). Using E[w_MST] = int_0^1 (E[C(p)]-1)dp
on this generalized-wheel graph and a renewal computation of rim-cycle arc
statistics gives, per vertex (m -> infinity limit, one ring):

  c(p) = (1-p)^3 (1 - x*p) / (1 - p(1-p)(1-x*p))

where x* is the limiting fraction of v-type vertices in a ring, itself
derived from the same ring-count recursion:

  x*(q) = (q - 3 - lambda(q)) / (lambda(q) + 1)

Predicted mu(q) = integral_0^1 c(p) dp.
"""
import numpy as np
from scipy.integrate import quad

MEASURED = {
    6: 0.18739, 7: 0.24557, 8: 0.25852, 9: 0.26548, 10: 0.26990,
    11: 0.27297, 12: 0.27532, 13: 0.27700, 14: 0.27837, 16: 0.28055,
    18: 0.28198, 20: 0.28310,
}


def lam(q):
    a = q - 4
    disc = a * a - 4
    if disc < 0:
        return 1.0
    return (a + np.sqrt(disc)) / 2


def xstar(q):
    L = lam(q)
    return (q - 3 - L) / (L + 1)


def c_of_p(p, x):
    return (1 - p) ** 3 * (1 - x * p) / (1 - p * (1 - p) * (1 - x * p))


def predicted_mu(q):
    x = xstar(q)
    val, err = quad(c_of_p, 0, 1, args=(x,))
    return val, x


print(f"{'q':>3} {'lambda':>8} {'x*':>8} {'predicted mu':>14} {'measured mu':>12} {'diff':>8}")
for q, meas in MEASURED.items():
    pred, x = predicted_mu(q)
    print(f"{q:3d} {lam(q):8.4f} {x:8.4f} {pred:14.5f} {meas:12.5f} {pred-meas:8.5f}")

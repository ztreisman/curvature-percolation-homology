"""
Exact (quenched) version of the ring-attachment calculation for mu(q),
replacing the i.i.d. Bernoulli(x*) treatment of the e/v-type pattern with
its true structure: a Sturmian word of slope alpha = x*(q) = 1/lambda(q).

Renewal sum, per vertex of one ring, at percolation parameter p:

  c(p) = (1-p) * sum_{L>=1} P(L) * E_k[ prod_{j in arc} (spoke-failure_j) ]

  P(L) = p^{L-1}(1-p)               (arc length via i.i.d. rim edges)
  spoke-failure_j = (1-p) if e-type, (1-p)^2 if V-type

For a Sturmian window of length L starting at a uniformly random phase,
the V-count is floor(L*alpha) with prob 1-{L*alpha}, floor(L*alpha)+1 with
prob {L*alpha}. So

  E_k[...] = (1-p)^L * (1-p)^floor(L alpha) * (1 - p*{L alpha})

versus the i.i.d. version (1-p)^L * (1 - alpha p)^L.

mu(q) = int_0^1 c(p) dp.
"""
import numpy as np
from scipy.integrate import quad

MEASURED = {
    7: 0.24557, 8: 0.25852, 9: 0.26548, 10: 0.26990,
    11: 0.27297, 12: 0.27532, 13: 0.27700, 14: 0.27837, 16: 0.28055,
    18: 0.28198, 20: 0.28310,
}


def lam(q):
    a = q - 4
    return (a + np.sqrt(a * a - 4)) / 2


def c_iid(p, alpha):
    return (1 - p) ** 3 * (1 - alpha * p) / (1 - p * (1 - p) * (1 - alpha * p))


def c_sturm(p, alpha, Lmax=400):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    L = np.arange(1, Lmax + 1)
    La = L * alpha
    fl = np.floor(La)
    fr = La - fl
    terms = p ** (L - 1) * (1 - p) ** (L + fl) * (1 - p * fr)
    return (1 - p) ** 2 * np.sum(terms)


def mu(q, model):
    alpha = 1.0 / lam(q)
    f = c_iid if model == "iid" else c_sturm
    val, _ = quad(lambda p: f(p, alpha), 0, 1, limit=200)
    return val


print(f"{'q':>3} {'measured':>9} {'iid':>9} {'d_iid':>8} {'sturm':>9} {'d_sturm':>8}")
for q, meas in MEASURED.items():
    a = mu(q, "iid")
    s = mu(q, "sturm")
    print(f"{q:3d} {meas:9.5f} {a:9.5f} {a-meas:+8.5f} {s:9.5f} {s-meas:+8.5f}")

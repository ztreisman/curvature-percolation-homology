"""
Full derivation of mu(q) = lim E[w_MST]/V on {3,q} disks, q >= 7.

  mu(q) = mu_wheel(alpha) - B_1 * alpha + O(alpha^2),   alpha = 1/lambda(q)

  mu_wheel(alpha) = int_0^1 c(p; alpha) dp
      c(p; alpha) = (1-p)^3 (1 - alpha p) / (1 - p(1-p)(1 - alpha p))
      [one ring attached to a contracted interior; v-type fraction alpha]

  B_1 = int_0^1 [ c(p; 0) - c_eff(p) ] dp
      c_eff(p)   = (1-pe)^2 (1-p) / (1 - pe (1-p))
      pe(p)      = p + (1-p) beta(p)              (effective rim connectivity)
      beta(p)    = [ p / (1-p+p^2) ]^2            (seam bridged via shared child)
      [inner-ring arcs merged through the ring outside them; the outermost
       ring, a (1-1/lambda) fraction of V, has no such correction]

Exact pieces: x*(q) = 1/lambda(q) (v-type fraction), the wheel density,
beta(p), pe(p), and the accounting c - c_eff (each verified separately).
"""
import numpy as np
from scipy.integrate import quad

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


def beta(p):
    return (p / (1 - p + p * p)) ** 2


def c_eff(p):
    pe = p + (1 - p) * beta(p)
    return (1 - pe) ** 2 * (1 - p) / (1 - pe * (1 - p))


def b(p):
    return c_wheel(p, 0.0) - c_eff(p)


mu_inf = quad(lambda p: c_wheel(p, 0.0), 0, 1)[0]
B1 = quad(b, 0, 1, limit=200)[0]
print(f"mu_wheel(alpha=0) = {mu_inf:.6f}   (3/2 - 2*pi/(3*sqrt 3) = {1.5 - 2*np.pi/(3*np.sqrt(3)):.6f})")
print(f"B_1 = int (c - c_eff) dp = {B1:.6f}")

print(f"\n{'q':>3} {'alpha':>7} {'mu_wheel':>9} {'-B1*alpha':>10} {'predicted':>10} {'measured':>9} {'diff':>8} {'diff*lam^2':>10}")
diffs = []
for q, m in MEASURED.items():
    a = 1 / lam(q)
    mw = quad(lambda p: c_wheel(p, a), 0, 1)[0]
    pred = mw - B1 * a
    diffs.append(pred - m)
    print(f"{q:3d} {a:7.4f} {mw:9.5f} {-B1*a:+10.5f} {pred:10.5f} {m:9.5f} {pred-m:+8.5f} {(pred-m)/a**2:+10.4f}")
diffs = np.array(diffs)
print(f"\nmax |diff| = {np.abs(diffs).max():.5f}, rms = {np.sqrt((diffs**2).mean()):.5f}")

# --- closed form attempt for B_1 via partial fractions (guarded) ---
try:
    import sympy as sp
    P = sp.symbols('p', real=True)
    D = 1 - P + P**2
    bet = (P / D)**2
    pe = P + (1 - P) * bet
    ceff = (1 - pe)**2 * (1 - P) / (1 - pe * (1 - P))
    cw = (1 - P)**3 / D
    integrand = sp.cancel(sp.together(cw - ceff))
    num, den = sp.fraction(integrand)
    print("\nintegrand numerator :", sp.factor(num))
    print("integrand denominator:", sp.factor(den))
    pf = sp.apart(integrand, P)
    F = sp.integrate(pf, P)
    val = sp.simplify(F.subs(P, 1) - F.subs(P, 0))
    print("closed form B_1     :", val)
    print("numeric of that     :", sp.N(val, 12))
except Exception as e:
    print("\nsympy closed-form attempt failed:", repr(e))

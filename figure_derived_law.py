"""
Derived law vs measurement, q >= 7. Prints LaTeX table rows and draws
figure_derived_law.png: measured mu(q) and H1/vertex against 1/lambda(q)
with the derived curves (no fitted parameters).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

MU = {7: 0.24557, 8: 0.25852, 9: 0.26548, 10: 0.26990, 11: 0.27297,
      12: 0.27532, 13: 0.27700, 14: 0.27837, 16: 0.28055, 18: 0.28198, 20: 0.28310}
H1 = {7: (0.0910, 0.0009), 8: (0.0755, 0.0005), 9: (0.0676, 0.0007),
      10: (0.0630, 0.0010), 11: (0.0594, 0.0004), 12: (0.0578, 0.0013),
      13: (0.0552, 0.0004), 14: (0.0535, 0.0008), 16: (0.0516, 0.0006),
      18: (0.0500, 0.0015), 20: (0.0488, 0.0016)}


def lam(q):
    a = q - 4
    return (a + np.sqrt(a * a - 4)) / 2


def c_wheel(p, a):
    return (1 - p) ** 3 * (1 - a * p) / (1 - p * (1 - p) * (1 - a * p))


def c_eff(p):
    beta = (p / (1 - p + p * p)) ** 2
    pe = p + (1 - p) * beta
    return (1 - pe) ** 2 * (1 - p) / (1 - pe * (1 - p))


def mu_wheel(a):
    return quad(lambda p: c_wheel(p, a), 0, 1)[0]


B1 = quad(lambda p: c_wheel(p, 0.0) - c_eff(p), 0, 1, limit=200)[0]
mu0 = mu_wheel(0.0)


def mu_pred(a):
    return mu_wheel(a) - B1 * a


def h1_pred(a):
    return mu_pred(a) - 0.25 + a / 4


print(f"mu_wheel(0) = {mu0:.5f} = 3/2 - 2pi/(3 sqrt3) = {1.5 - 2*np.pi/(3*np.sqrt(3)):.5f}")
print(f"B1 = {B1:.5f}")
print(f"H1 intercept 5/4 - 2pi/(3 sqrt3) = {1.25 - 2*np.pi/(3*np.sqrt(3)):.5f}")

print("\n--- mu table rows (q & lambda & measured & predicted & diff) ---")
mu_d = []
for q, m in MU.items():
    a = 1 / lam(q)
    pm = mu_pred(a)
    mu_d.append(pm - m)
    print(f"{q} & {lam(q):.3f} & {m:.5f} & {pm:.5f} & ${pm-m:+.5f}$ \\\\")
print(f"rms {np.sqrt(np.mean(np.square(mu_d))):.5f}, max {np.max(np.abs(mu_d)):.5f}")

print("\n--- H1 table rows (q & measured +- & predicted & diff) ---")
h_d = []
for q, (m, e) in H1.items():
    a = 1 / lam(q)
    ph = h1_pred(a)
    h_d.append(ph - m)
    print(f"{q} & ${m:.4f}\\pm{e:.4f}$ & {ph:.4f} & ${ph-m:+.4f}$ \\\\")
print(f"rms {np.sqrt(np.mean(np.square(h_d))):.5f}, max {np.max(np.abs(h_d)):.5f}")

# effective slope of the derived H1 law over the tested range, for comparison with the fit
qs = np.array(list(H1))
al = 1 / np.array([lam(q) for q in qs])
hp = np.array([h1_pred(a) for a in al])
slope, intercept = np.polyfit(al, hp, 1)
print(f"\nlinear fit to the *derived* H1 curve over q=7..20: {intercept:.4f} + {slope:.4f}/lambda")

# ---- figure ----
aa = np.linspace(0.0, 0.40, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
ax.plot(aa, [mu_pred(a) for a in aa], "k-", lw=1.6, label="derived: $\\mu_{\\rm wheel}(1/\\lambda) - B_1/\\lambda$")
ax.plot(aa, [mu_wheel(a) for a in aa], "k--", lw=1.0, label="one-ring term only")
ax.plot(al, [MU[q] for q in qs], "o", color="#d62728", ms=6, label="measured $\\mu(q)$")
for q, a in zip(qs, al):
    ax.annotate(str(q), (a, MU[q]), textcoords="offset points", xytext=(4, -10), fontsize=8)
ax.set_xlabel("$1/\\lambda(q)$")
ax.set_ylabel("$\\mu(q) = \\lim E[w_{\\rm MST}]/V$")
ax.set_title("bulk MST density")
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
ax.plot(aa, [h1_pred(a) for a in aa], "k-", lw=1.6, label="derived, no free parameters")
ax.errorbar(al, [H1[q][0] for q in qs], yerr=[H1[q][1] for q in qs], fmt="o",
            color="#d62728", ms=6, capsize=2, label="measured $H_1$/vertex")
for q, a in zip(qs, al):
    ax.annotate(str(q), (a, H1[q][0]), textcoords="offset points", xytext=(4, -10), fontsize=8)
ax.set_xlabel("$1/\\lambda(q)$")
ax.set_ylabel("total $H_1$ persistence per vertex")
ax.set_title("loop content")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("figure_derived_law.png", dpi=220)
print("saved figure_derived_law.png")

"""Numerical data for the figures and the numbers quoted in Lecture 8 (iteration complexity).

The example is the worked example of Lecture 5, depth-one QAOA for MaxCut on the 4-cycle, in the
conventions of code/ch05_worked_example.py and code/ch07_figures.py: the cost Hamiltonian is minus
the cut, gamma is the angle of the four gates R_Z (between CNOTs) of the edges, 2 beta that of the
four gates R_X of the mixer, and F(beta, gamma) = -2 + sin 4 beta sin 2 gamma. Stochastic gradient
descent starts at the point (beta, gamma) = (0.1, -0.1) of the worked example and uses the
parameter-shift rule gate by gate, 16 circuits for each gradient, with M copies of each circuit.

Writes, into figures/ch08/:
  sgd_unbiased.dat  k and, for M = 10, 100, 1000 copies per circuit, the running average
                    (1/k) sum_{j<k} E|grad F(theta_j)|^2 over the runs (avgM) and the bound of the
                    theorem on stochastic gradient descent (bndM)
  sgd_biased.dat    the same for a device whose mixer gates are all over-rotated by delta
  floors.dat        the bias b and the floors of the parameter-shift rule, of simultaneous
                    perturbation and of the two-point estimator, with the best perturbations
and prints the numbers quoted in the text, after checking the formulas of the lecture numerically.
"""
import os
import numpy as np

OUT = "figures/ch08"
os.makedirs(OUT, exist_ok=True)
n, edges = 4, [(0, 1), (1, 2), (2, 3), (3, 0)]
rng = np.random.default_rng(2028)

# --------------------------------------------------------------------------- the circuit, vectorised over runs
bits = np.array([[(x >> (n - 1 - q)) & 1 for q in range(n)] for x in range(2**n)])
z = 1 - 2 * bits
zz = np.array([z[:, i] * z[:, j] for i, j in edges])           # (4 edges, 16 strings)
cuts = np.array([sum(b[i] != b[j] for i, j in edges) for b in bits])
vals = -cuts.astype(float)                                      # the measured value of H_C
cutvals = np.array([0.0, -2.0, -4.0])                           # the values for cuts 0, 2, 4
onehot = np.array([[c == k for k in (0, 2, 4)] for c in cuts], float)

def probs(tz, tx):
    """Probabilities of the 16 bit strings; tz, tx of shape (R, 4): the angles of the R_Z gates
    of the edges (CNOT-R_Z-CNOT, that is exp(-i t Z_i Z_j / 2)) and of the R_X gates of the mixer."""
    R = tz.shape[0]
    psi = (np.exp(-0.5j * tz @ zz) / 4).reshape((R,) + (2,) * n)
    for q in range(n):
        c, s = np.cos(tx[:, q] / 2), -1j * np.sin(tx[:, q] / 2)
        U = np.stack([np.stack([c, s], -1), np.stack([s, c], -1)], -2)
        m = np.moveaxis(psi, q + 1, -1)
        psi = np.moveaxis(np.einsum("rab,r...b->r...a", U, m), -1, q + 1)
    return (np.abs(psi)**2).reshape(R, -1)

# a dense check of the circuit against Kronecker products of the gates
def dense(tz, tx):
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    def kron(*ms):
        out = np.eye(1)
        for m in ms: out = np.kron(out, m)
        return out
    psi = kron(*[H1] * n) @ np.eye(2**n)[0]
    for (i, j), t in zip(edges, tz):
        psi = np.exp(-0.5j * t * z[:, i] * z[:, j]) * psi
    rx = lambda t: np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])
    psi = kron(*[rx(t) for t in tx]) @ psi
    return np.abs(psi)**2
for _ in range(3):
    tz_, tx_ = rng.uniform(-3, 3, 4), rng.uniform(-3, 3, 4)
    assert np.allclose(probs(tz_[None], tx_[None])[0], dense(tz_, tx_))

def F_closed(beta, gamma): return -2 + np.sin(4 * beta) * np.sin(2 * gamma)
def grad_closed(beta, gamma):
    return np.array([4 * np.cos(4 * beta) * np.sin(2 * gamma), 2 * np.sin(4 * beta) * np.cos(2 * gamma)])
def angles(beta, gamma, delta=0.0):
    beta, gamma = np.atleast_1d(beta), np.atleast_1d(gamma)
    return np.repeat(gamma[:, None], 4, 1), np.repeat(2 * beta[:, None], 4, 1) + delta
for b_, g_ in [(0.1, -0.1), (0.7, -0.4), (0.2, -0.3)]:
    assert abs(probs(*angles(b_, g_)) @ vals - F_closed(b_, g_)) < 1e-12

beta0, gamma0 = 0.1, -0.1
F0, Fstar = F_closed(beta0, gamma0), -3.0
print(f"worked example: F(0.1, -0.1) = {F0:.5f}, minimum -3 at (pi/8, -pi/4); C(theta_0) - C* = {F0 - Fstar:.5f}")

# --------------------------------------------------------------------------- bounds on the derivatives
gb, gg = np.meshgrid(np.linspace(-np.pi, np.pi, 721), np.linspace(-np.pi, np.pi, 721))
Hbb = -16 * np.sin(4 * gb) * np.sin(2 * gg); Hgg = -4 * np.sin(4 * gb) * np.sin(2 * gg)
Hbg = 8 * np.cos(4 * gb) * np.cos(2 * gg)
lam = np.abs((Hbb + Hgg) / 2) + np.sqrt(((Hbb - Hgg) / 2)**2 + Hbg**2)
gn = np.hypot(4 * np.cos(4 * gb) * np.sin(2 * gg), 2 * np.sin(4 * gb) * np.cos(2 * gg))
print(f"smoothness of F: max |grad F| = {gn.max():.4f} (bound sqrt(20) = {np.sqrt(20):.4f}); "
      f"max |Hessian| = {lam.max():.4f} (bound 20)")
assert gn.max() <= np.sqrt(20) + 1e-12 and lam.max() <= 20 + 1e-12

# the cost as a function of the 8 gate angles: entries of the gradient and Hessian at most w/2 = 2
# (by the shift rule), the Hessian at most d w / 2 = 16 in norm
def C_gates(t):                                   # t of shape (R, 8): 4 edge angles, then 4 mixer angles
    return probs(t[:, :4], t[:, 4:]) @ vals
P = rng.uniform(-np.pi, np.pi, (300, 8))
I8 = np.eye(8) * np.pi / 2
Hmax_entry, Hmax_norm, gmax = 0, 0, 0
H = np.zeros((300, 8, 8))
for j in range(8):
    for k in range(8):
        H[:, j, k] = (C_gates(P + I8[j] + I8[k]) - C_gates(P + I8[j] - I8[k])
                      - C_gates(P - I8[j] + I8[k]) + C_gates(P - I8[j] - I8[k])) / 4
    gmax = max(gmax, np.max(np.abs(C_gates(P + I8[j]) - C_gates(P - I8[j])) / 2))
Hmax_entry = np.max(np.abs(H)); Hmax_norm = np.max(np.abs(np.linalg.eigvalsh(H)))
print(f"cost of the 8 gate angles at 300 random points: max |partial| = {gmax:.4f}, max |Hessian entry| = "
      f"{Hmax_entry:.4f} (bound 2), max |Hessian| = {Hmax_norm:.4f} (bound 16)")
assert gmax <= 2 + 1e-12 and Hmax_entry <= 2 + 1e-12 and Hmax_norm <= 16

# --------------------------------------------------------------------------- the descent lemma on one gate
# f(phi) = a + A sin(phi - 0.1) for the R_Z gate of edge 12 (Lecture 7); L = A; the minimiser 0.1 - pi/2
a1, A1 = -2.03868, 0.19471
f1 = lambda p: a1 + A1 * np.sin(p - 0.1)
df1 = lambda p: A1 * np.cos(p - 0.1)
phi = [-0.1]
for _ in range(3):
    phi.append(phi[-1] - df1(phi[-1]) / A1)
q1 = f1(phi[0]) - df1(phi[0])**2 / (2 * A1)
print(f"one gate: f(-0.1) = {f1(-0.1):.4f}, f' = {df1(-0.1):.4f}; step 1/L = {1/A1:.3f} to phi_1 = {phi[1]:.4f}; "
      f"the parabola there {q1:.4f}, f(phi_1) = {f1(phi[1]):.4f}; guaranteed decrease {df1(-0.1)**2/(2*A1):.4f}, "
      f"actual {f1(-0.1) - f1(phi[1]):.4f}")
print(f"  errors phi_k - phi*: {[f'{p - (0.1 - np.pi/2):.3g}' for p in phi]}")
# a check of the descent lemma for the parabola
pp = np.linspace(-np.pi, np.pi, 2001)
assert np.all(f1(pp) <= f1(-0.1) + df1(-0.1) * (pp + 0.1) + A1 / 2 * (pp + 0.1)**2 + 1e-12)

# --------------------------------------------------------------------------- sources of bias
# the Bell state on ibmq_santiago, 1000 shots (Kungurtsev et al., Fig. 9)
pb = {"00": 0.423, "01": 0.012, "10": 0.010, "11": 0.555}
zz_b = pb["00"] + pb["11"] - pb["01"] - pb["10"]
z_a, z_b = pb["00"] + pb["01"] - pb["10"] - pb["11"], pb["00"] + pb["10"] - pb["01"] - pb["11"]
print(f"Bell state on ibmq_santiago: <ZZ> = {zz_b:.3f} (ideal 1, standard error {np.sqrt((1 - zz_b**2)/1000):.3f}), "
      f"<Z> = {z_a:.3f}, {z_b:.3f} (ideal 0, standard error {np.sqrt((1 - z_a**2)/1000):.3f})")

# read-out errors: independent flips with the rates of four IBM devices, one for each qubit
rates = {"lima": (0.0078, 0.0260), "santiago": (0.0274, 0.0448), "bogota": (0.0496, 0.1916), "manila": (0.0114, 0.034)}
A_ro = np.eye(1)
for (q0, q1) in rates.values():                  # P(reported | true) for one qubit, rows reported
    A_ro = np.kron(A_ro, np.array([[1 - q0, q1], [q0, 1 - q1]]))
for b_, g_ in [(0.1, -0.1), (0.3, -0.6), (np.pi / 8, -np.pi / 4)]:
    p = probs(*angles(b_, g_))[0]
    pr = A_ro @ p
    zq = [p @ z[:, q] for q in range(n)]
    assert np.max(np.abs(zq)) < 1e-12          # <Z_i> = 0 by the symmetry X^{(x)n}
    av = [1 - q0 - q1 for (q0, q1) in rates.values()]; cv = [q1 - q0 for (q0, q1) in rates.values()]
    pred = -2 + sum(av[i] * av[j] for i, j in edges) * (F_closed(b_, g_) + 2) / 4 + sum(cv[i] * cv[j] for i, j in edges) / 2
    assert abs(pr @ vals - pred) < 1e-12
sa = sum((1 - rates[k1][0] - rates[k1][1]) * (1 - rates[k2][0] - rates[k2][1])
         for k1, k2 in [("lima", "santiago"), ("santiago", "bogota"), ("bogota", "manila"), ("manila", "lima")])
print(f"read-out with the rates of lima, santiago, bogota, manila on the four qubits: measured cost "
      f"= -2 + {sa/4:.4f} (F + 2) + const; the minimiser does not move")

# a miscalibrated mixer: every R_X(2 beta) applied as R_X(2 beta + delta)
delta = 0.05
dev_t = np.array([[0.3, -0.5], [0.12, -0.9], [0.39, -0.78]])
for b_, g_ in dev_t:
    assert abs(probs(*angles(b_, g_, delta)) @ vals - F_closed(b_ + delta / 2, g_)) < 1e-12
Pg = rng.uniform(-np.pi, np.pi, (20000, 8))
Pd = Pg.copy(); Pd[:, 4:] += delta
bias_gates = np.max(np.abs(C_gates(Pd) - C_gates(Pg)))
zeta = 8 * np.sin(delta)
print(f"miscalibrated mixer, delta = {delta}: sup |C~ - C| over 20000 random gate angles = {bias_gates:.4f}, "
      f"bound by the lemma on gate errors 4 * 4 * 2 sin(delta/4) = {16 * 2 * np.sin(delta / 4):.4f}; "
      f"on the QAOA angles 2 sin(delta) = {2 * np.sin(delta):.4f}")
print(f"  zeta = sup |grad F(beta + delta/2, gamma) - grad F| = 8 sin(delta) = {zeta:.4f}, zeta^2 = {zeta**2:.4f}; "
      f"|grad F|^2 at the minimiser of the device's cost = 16 sin^2(2 delta) = {16 * np.sin(2 * delta)**2:.4f}")
assert bias_gates <= 32 * np.sin(delta / 4)
dz = np.hypot(*(grad_closed(gb + delta / 2, gg) - grad_closed(gb, gg)))
assert abs(dz.max() - zeta) < 1e-3

# --------------------------------------------------------------------------- stochastic gradient descent
def sample_mean(pc, M):
    """Means of M measured values with the probabilities pc (R, 3) of the cuts 0, 2, 4."""
    pc = np.clip(pc, 0, 1)
    n0 = rng.binomial(M, pc[:, 0])
    r1 = np.where(pc[:, 0] < 1, pc[:, 1] / np.maximum(1 - pc[:, 0], 1e-300), 0)
    n1 = rng.binomial(M - n0, np.clip(r1, 0, 1))
    n2 = M - n0 - n1
    return (cutvals[1] * n1 + cutvals[2] * n2) / M

S = np.pi / 2
def psr_gradient(beta, gamma, M, delta=0.0, exact=False):
    """The gradient of F estimated gate by gate with M copies of each of the 16 circuits."""
    tz, tx = angles(beta, gamma, delta)
    g = np.zeros((len(beta), 2))
    for e in range(4):
        for sg in (1, -1):
            t = tz.copy(); t[:, e] += sg * S
            pc = probs(t, tx) @ onehot
            m = pc @ cutvals if exact else sample_mean(pc, M)
            g[:, 1] += sg * m / 2
    for q in range(4):
        for sg in (1, -1):
            t = tx.copy(); t[:, q] += sg * S
            pc = probs(tz, t) @ onehot
            m = pc @ cutvals if exact else sample_mean(pc, M)
            g[:, 0] += 2 * sg * m / 2
    return g

bb = rng.uniform(-1, 1, 50); gg_ = rng.uniform(-1, 1, 50)
assert np.allclose(psr_gradient(bb, gg_, 1, exact=True), grad_closed(bb, gg_).T)
assert np.allclose(psr_gradient(bb, gg_, 1, delta, exact=True), grad_closed(bb + delta / 2, gg_).T)

# the variance of the estimate: V(theta) = sum over gates of s^2 (sigma_+^2 + sigma_-^2) / (4M)
def var_sum(beta, gamma, delta=0.0):
    tz, tx = angles(beta, gamma, delta)
    v = np.zeros(len(beta))
    for e in range(4):
        for sg in (1, -1):
            t = tz.copy(); t[:, e] += sg * S
            p = probs(t, tx); mu = p @ vals
            v += (p @ vals**2 - mu**2) / 4
    for q in range(4):
        for sg in (1, -1):
            t = tx.copy(); t[:, q] += sg * S
            p = probs(tz, t); mu = p @ vals
            v += 4 * (p @ vals**2 - mu**2) / 4
    return v                                      # M V
Bg, Gg = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 121), np.linspace(-np.pi / 2, np.pi / 2, 121))
MV = var_sum(Bg.ravel(), Gg.ravel()).max()
MVd = var_sum(Bg.ravel(), Gg.ravel(), delta).max()
sig2 = max(np.max(probs(*angles(Bg.ravel(), Gg.ravel())) @ vals**2 - (probs(*angles(Bg.ravel(), Gg.ravel())) @ vals)**2), 0)
print(f"variance: one measured value at most {sig2:.4f} on the grid (at most 4 for values in [-4, 0]); "
      f"M V <= {MV:.4f} (and {MVd:.4f} for the miscalibrated device), against 10 sigma^2 <= 40")

alpha, Lb, K, runs = 0.05, 20.0, 3000, 200
Ms = [10, 100, 1000]
def run_sgd(M, delta=0.0):
    th = np.tile([beta0, gamma0], (runs, 1)).astype(float)
    g2 = np.zeros(K)
    for k in range(K):
        g2[k] = np.mean(np.sum(grad_closed(th[:, 0], th[:, 1])**2, axis=0))
        th -= alpha * psr_gradient(th[:, 0], th[:, 1], M, delta)
    return np.cumsum(g2) / np.arange(1, K + 1), g2

ks = np.unique(np.round(np.exp(np.linspace(0, np.log(K), 70))).astype(int))
for name, dl in (("unbiased", 0.0), ("biased", delta)):
    cols, last = [], []
    mv = MV if dl == 0 else MVd
    for M in Ms:
        avg, g2 = run_sgd(M, dl)
        bnd = 2 * (F0 - Fstar) / (alpha * np.arange(1, K + 1)) + alpha * Lb * mv / M + (zeta**2 if dl else 0)
        assert np.all(avg <= bnd)
        cols.append((avg, bnd)); last.append((M, avg[-1], np.mean(g2[-500:]), bnd[-1]))
    with open(f"{OUT}/sgd_{name}.dat", "w") as fh:
        fh.write("k " + " ".join(f"avg{M} bnd{M}" for M in Ms) + "\n")
        for k in ks:
            fh.write(f"{k} " + " ".join(f"{a[k-1]:.6e} {b[k-1]:.6e}" for a, b in cols) + "\n")
    for M, a_, l_, b_ in last:
        print(f"SGD {name}, M = {M}: running average after {K} steps {a_:.4f}, mean |grad F|^2 over the last 500 "
              f"steps {l_:.4f}, bound {b_:.4f} (alpha L V = {alpha * Lb * mv / M:.4f})")
# exact gradient descent from the same point, for comparison
th = np.array([[beta0, gamma0]])
for k in range(200):
    th = th - alpha * grad_closed(th[:, 0], th[:, 1]).T
print(f"gradient descent, alpha = 0.05, after 200 steps: {th[0]}, F = {F_closed(*th[0]):.6f}; "
      f"bound 2 L (C0 - C*) / K for K = 100: {2 * Lb * (F0 - Fstar) / 100:.4f}")

# --------------------------------------------------------------------------- the derivative-free estimators under bias
# bias function b(theta) = bb sin(3 beta + 5 gamma + 1), noise of variance s2 for each value
d, Lc, L3 = 2, 20.0, 20.0**1.5
bb_, s2, cc = 0.01, 0.004, 0.1
th0 = np.array([0.2, -0.3])
Ft = lambda t: F_closed(t[..., 0], t[..., 1]) + bb_ * np.sin(3 * t[..., 0] + 5 * t[..., 1] + 1)
gr = grad_closed(*th0)
Ds = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
gs = np.array([(Ft(th0 + cc * D) - Ft(th0 - cc * D)) / (2 * cc) * D for D in Ds])
bias_spsa = np.linalg.norm(gs.mean(0) - gr)
m2_spsa = np.mean(np.sum(gs**2, 1)) + d * 2 * s2 / (4 * cc**2)
print(f"SPSA under bias at (0.2, -0.3), c = {cc}, b = {bb_}: |E g - grad F| = {bias_spsa:.4f} <= "
      f"c^2 L3 d^1.5 / 6 + b / c = {cc**2 * L3 * d**1.5 / 6 + bb_ / cc:.4f}; E|g|^2 = {m2_spsa:.4f} <= "
      f"{4 * d * gr @ gr + Lc**2 * cc**2 * d**3 + d * (4 * bb_**2 + s2) / (2 * cc**2):.4f}")
assert bias_spsa <= cc**2 * L3 * d**1.5 / 6 + bb_ / cc
assert m2_spsa <= 4 * d * gr @ gr + Lc**2 * cc**2 * d**3 + d * (4 * bb_**2 + s2) / (2 * cc**2)
U = rng.normal(size=(2000000, 2))
v2 = (Ft(th0 + cc * U) - Ft(th0)) / cc
g2p = v2[:, None] * U
gradc = np.exp(-10 * cc**2) * gr
bias_2p = np.linalg.norm(g2p.mean(0) - gradc)
m2_2p = np.mean(np.sum(g2p**2, 1)) + 2 * s2 * d / cc**2
print(f"two-point under bias, c = {cc}: |E g - grad F_c| = {bias_2p:.4f} (Monte Carlo error about "
      f"{np.sqrt(np.var(g2p, 0).sum() / len(U)):.4f}) <= 2 sqrt(2/pi) b / c = {2 * np.sqrt(2 / np.pi) * bb_ / cc:.4f}; "
      f"E|g|^2 = {m2_2p:.3f} <= {4 * (d + 4) * gr @ gr + cc**2 * Lc**2 * (d + 6)**3 + 2 * d * (4 * bb_**2 + s2) / cc**2:.3f}")
assert m2_2p <= 4 * (d + 4) * gr @ gr + cc**2 * Lc**2 * (d + 6)**3 + 2 * d * (4 * bb_**2 + s2) / cc**2

# the floors as functions of b, for a circuit of d = 8 Pauli rotations with |O| = 1, so that
# W = 1, L = d and L3 = d^(3/2) by the lemma on derivatives, and max |grad C|^2 <= d
d, Lc, L3 = 8, 8.0, 8.0**1.5
kap = L3 * d**1.5 / 6
def floors(b_):
    c_s = (b_ / (2 * kap))**(1 / 3)
    zs = kap * c_s**2 + b_ / c_s
    c_t = ((64 / np.pi) * b_**2 / (Lc**2 * (d + 3)**3))**0.25
    tp = (32 / np.pi) * b_**2 / c_t**2 + c_t**2 / 2 * Lc**2 * (d + 3)**3
    assert abs(tp - 8 / np.sqrt(np.pi) * b_ * Lc * (d + 3)**1.5) < 1e-9 * max(1, tp)
    assert abs(2 * zs**2 - 9 / 2**(1 / 3) * kap**(2 / 3) * b_**(4 / 3)) < 1e-9 * max(1, zs**2)
    cg = np.exp(np.linspace(-12, 2, 20001))     # the minima over c, by brute force
    assert 2 * zs**2 <= np.min(2 * (kap * cg**2 + b_ / cg)**2) * (1 + 1e-6)
    assert tp <= np.min((32 / np.pi) * b_**2 / cg**2 + cg**2 / 2 * Lc**2 * (d + 3)**3) * (1 + 1e-6)
    return d * b_**2, 2 * zs**2, tp, c_s, c_t
bs = np.exp(np.linspace(np.log(1e-5), np.log(1e-1), 41))
with open(f"{OUT}/floors.dat", "w") as fh:
    fh.write("b psr spsa twopoint cspsa ctwopoint\n")
    for b_ in bs:
        fh.write(f"{b_:.6e} " + " ".join(f"{v:.6e}" for v in floors(b_)) + "\n")
print(f"floors for d = {d} Pauli rotations, |O| = 1: kappa = d^3/6 = {kap:.2f}; SPSA {9 / 2**(1/3) * kap**(2/3):.1f} b^(4/3); "
      f"two-point {8 / np.sqrt(np.pi) * Lc * (d + 3)**1.5:.0f} b")
for b_ in (1e-4, 1e-3, 1e-2):
    p_, s_, t_, cs_, ct_ = floors(b_)
    print(f"  b = {b_}: parameter shift {p_:.2e}; SPSA {s_:.3e} at c = {cs_:.4f}; two-point {t_:.3e} at c = {ct_:.4f}")

# --------------------------------------------------------------------------- the Polyak-Lojasiewicz inequality
# the sinusoid: (1/2) f'^2 = (A/2)(1 + cos e)(f - f*); and x^2 + 3 sin^2 x (Karimi et al.: mu = 1/32)
e = np.linspace(-3, 3, 6001)
assert np.allclose(0.5 * (A1 * np.sin(e))**2, A1 / 2 * (1 + np.cos(e)) * A1 * (1 - np.cos(e)))
x = np.linspace(-50, 50, 2000001); x = x[np.abs(x) > 1e-6]
ratio = 0.5 * (2 * x + 3 * np.sin(2 * x))**2 / (x**2 + 3 * np.sin(x)**2)
print(f"x^2 + 3 sin^2 x: least (1/2) f'^2 / f = {ratio.min():.4f} at x = {x[np.argmin(ratio)]:.3f} (>= 1/32); "
      f"least second derivative {np.min(2 + 6 * np.cos(2 * x)):.2f}, so not convex")
assert ratio.min() >= 1 / 32

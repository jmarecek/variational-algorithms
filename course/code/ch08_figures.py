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
  pl_gd.dat         k, F(theta_k) - F* for gradient descent with the step 1/20, the guarantee of the
                    Polyak-Lojasiewicz inequality on the sublevel set of the start, and the local rate 0.64^k
  qaoa_constants.dat  p and, for the 4-cycle and the 3-regular graph of Lecture 4, the largest norms of the
                    gradient and of the Hessian of QAOA of depth p found by a local search, with the bounds
  twocost_band.dat  the band -cos(theta) -+ b of the two costs of the lower bound, as a closed polygon
  sgd_steps.dat     the step and, for M = 100 and K = 1000, the average over the runs, the two terms of the
                    bound and their sum
  decr_unbiased.dat, decr_biased.dat   k and the mean of |grad F(theta_k)|^2 over the runs for the shift rule
                    with a constant step, with decreasing gains, and for simultaneous perturbation with
                    decreasing gains, M = 100
  dfo_unbiased.dat, dfo_biased.dat   circuits and the running averages for the shift rule, simultaneous
                    perturbation with its step and with 1/20, and the two-point estimator, M = 100, with the
                    bounds of the shift rule and of simultaneous perturbation
  descent_plane.dat  beta and the level curves of F near the start, for the picture of the descent lemma
  gd_levels.dat     closed level curves of F in the cell of the minimiser (pi/8, -pi/4), the first that of F(theta_0),
                    then F = -2.25, -2.5, -2.75, -2.9, -2.99, -2.97, -2.94
  gd_path.dat       k, the iterates of gradient descent with the step 1/20, F, |grad F|^2, F - |grad F|^2 / (2L)
                    and the step -grad F / L
  pl_ratio.dat      beta, gamma and log10 of (1/2)|grad F|^2 / (F - F*) on the cell
  noise_estimates.dat  200 estimates of grad F(theta_0) by the shift rule on the over-rotated device, M = 1000
  sgd_cloud_unbiased.dat, sgd_cloud_biased.dat   the iterates of all the runs of stochastic gradient descent after
                    3000 steps, for M = 10, 100, 1000
  sgd_steps_paths.dat  k and the path of the first run for the steps 0.002, 0.027 and 0.1, M = 100
  decr_cloud_unbiased.dat, decr_cloud_biased.dat   the iterates of all the runs with the constant step at k = 10^4,
                    with decreasing gains at k = 100, 1000, 10^4, and of simultaneous perturbation at k = 10^4
  dfo_paths_unbiased.dat, dfo_paths_biased.dat   circuits and the path of the first run of each estimator, up to 8000
  dfo_cloud_unbiased.dat, dfo_cloud_biased.dat   the iterates of all the runs of each estimator after 48000 circuits
  noise_model.dat   beta, C, C~ and C -+ b on the line gamma = -pi/4, for the device whose mixer is over-rotated by delta
  noise_model_est.dat  one estimate with M = 1000 copies at each of 33 angles on that line
  noise_model_bars.dat  M, beta = 0.1, C~, the standard deviation of an estimate with M copies, and C
  gd_estimators.dat  k and the paths of simultaneous perturbation (step 1/40) and of the two-point estimator (step 1/80)
                    with exact values and c = 0.1, from theta_0
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
    return np.cumsum(g2) / np.arange(1, K + 1), g2, th

ks = np.unique(np.round(np.exp(np.linspace(0, np.log(K), 70))).astype(int))
for name, dl in (("unbiased", 0.0), ("biased", delta)):
    cols, last, clouds = [], [], []
    mv = MV if dl == 0 else MVd
    for M in Ms:
        avg, g2, thK = run_sgd(M, dl)
        clouds.append(thK)
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
    # the iterates of all the runs after K steps, about the minimiser of the cost that the device computes
    with open(f"{OUT}/sgd_cloud_{name}.dat", "w") as fh:
        fh.write(" ".join(f"b{M} g{M}" for M in Ms) + "\n")
        for r in range(runs):
            fh.write(" ".join(f"{c_[r, 0]:.6f} {c_[r, 1]:.6f}" for c_ in clouds) + "\n")
    for M, c_ in zip(Ms, clouds):
        ctr = c_.mean(0)
        print(f"  the {runs} iterates after {K} steps, M = {M}: centre ({ctr[0]:.4f}, {ctr[1]:.4f}), root-mean-square spread "
              f"{np.std(c_[:, 0]):.4f} along beta and {np.std(c_[:, 1]):.4f} along gamma; mean |grad F|^2 over them "
              f"{np.mean(np.sum(grad_closed(c_[:, 0], c_[:, 1])**2, axis=0)):.4f}")
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

# =========================================================================== the figures added later
rng = np.random.default_rng(2032)       # a generator of its own from here on, so that the numbers above do not change

# the estimators of the simulations below, with all the circuits of an iteration in one batch
def F_est(beta, gamma, M, dl=0.0):
    return sample_mean(probs(*angles(beta, gamma, dl)) @ onehot, M)
def psr_fast(beta, gamma, M, dl=0.0, exact=False):
    """The same estimate as psr_gradient, with the 16 shifted circuits of all the runs in one batch."""
    tz, tx = angles(beta, gamma, dl)
    TZ, TX = np.repeat(tz[None], 16, 0), np.repeat(tx[None], 16, 0)
    for e in range(4):
        TZ[2 * e, :, e] += S; TZ[2 * e + 1, :, e] -= S
        TX[8 + 2 * e, :, e] += S; TX[9 + 2 * e, :, e] -= S
    pc = probs(TZ.reshape(-1, 4), TX.reshape(-1, 4)) @ onehot
    m = (pc @ cutvals if exact else sample_mean(pc, M)).reshape(16, len(beta))
    return np.stack([(m[8::2] - m[9::2]).sum(0), (m[0:8:2] - m[1:8:2]).sum(0) / 2], 1)
assert np.allclose(psr_fast(bb, gg_, 1, exact=True), grad_closed(bb, gg_).T)
assert np.allclose(psr_fast(bb, gg_, 1, delta, exact=True), grad_closed(bb + delta / 2, gg_).T)
def spsa_grad(th, c_, M, dl=0.0):
    D = rng.choice([-1.0, 1.0], size=th.shape)
    y = F_est(*np.vstack([th + c_ * D, th - c_ * D]).T, M, dl).reshape(2, -1)
    return ((y[0] - y[1]) / (2 * c_))[:, None] * D
def twopoint_grad(th, c_, M, dl=0.0):
    u = rng.normal(size=th.shape)
    y = F_est(*np.vstack([th + c_ * u, th]).T, M, dl).reshape(2, -1)
    return ((y[0] - y[1]) / c_)[:, None] * u

# --------------------------------------------------------------------------- the descent lemma: two steps
q2 = f1(phi[1]) - df1(phi[1])**2 / (2 * A1)
print(f"descent lemma, second step: f'(phi_1) = {df1(phi[1]):.4f}, phi_2 = {phi[2]:.4f}, f(phi_2) = {f1(phi[2]):.5f}; "
      f"the vertex of the parabola at phi_1 {q2:.5f}; guaranteed decrease {df1(phi[1])**2 / (2 * A1):.4f}, "
      f"actual {f1(phi[1]) - f1(phi[2]):.4f}")
q1v = f1(phi[0]) - df1(phi[0])**2 / (2 * A1)                  # (q1 above was reused by the read-out rates)
print(f"  at phi_1: the parabola {q1v:.4f}, f {f1(phi[1]):.4f}, the tangent at phi_0 {f1(-0.1) + df1(-0.1) * (phi[1] + 0.1):.4f}")

# --------------------------------------------------------------------------- the Bell state: estimates and intervals
for M_ in (1000, 100000):
    print(f"Bell state with {M_} copies: <ZZ> = {zz_b:.3f} +- {2 * np.sqrt((1 - zz_b**2) / M_):.4f}, "
          f"<Z> = {z_a:.3f} +- {2 * np.sqrt((1 - z_a**2) / M_):.4f}, {z_b:.3f} +- {2 * np.sqrt((1 - z_b**2) / M_):.4f} "
          f"(two standard errors)")
for o_, p_ in pb.items():
    print(f"  outcome {o_}: frequency {p_:.3f}, standard error {np.sqrt(p_ * (1 - p_) / 1000):.4f}")

# --------------------------------------------------------------------------- the Polyak-Lojasiewicz inequality
# (a) the sinusoid of one gate: (1/2) f'^2 >= mu (f - f*) on |e| <= e0, with mu = A (1 + cos e0) / 2
e0 = 2.0
mu_s = A1 * (1 + np.cos(e0)) / 2
ee = np.linspace(-np.pi, np.pi, 4001)
lhs, rhs = 0.5 * (A1 * np.sin(ee))**2, mu_s * A1 * (1 - np.cos(ee))
assert np.all(lhs[np.abs(ee) <= e0] >= rhs[np.abs(ee) <= e0] - 1e-15) and np.all(lhs[np.abs(ee) > e0 + 1e-3] < rhs[np.abs(ee) > e0 + 1e-3])
print(f"PL on one gate: A = {A1}, e0 = {e0}: mu = A (1 + cos e0)/2 = {mu_s:.4f}; at e = pi, (1/2) f'^2 = 0 and f - f* = 2A = {2 * A1:.4f}")
# (b) the worked example: mu on the sublevel set {F <= F(0.1, -0.1)} of the minimiser (pi/8, -pi/4), in the cell
#     0 < beta < pi/4, -pi/2 < gamma < 0, and gradient descent with the step 1/20 from (0.1, -0.1)
Bc, Gc = np.meshgrid(np.linspace(1e-4, np.pi / 4 - 1e-4, 1501), np.linspace(-np.pi / 2 + 1e-4, -1e-4, 1501))
Fc = F_closed(Bc, Gc); gc2 = np.sum(grad_closed(Bc, Gc)**2, axis=0)
inS = (Fc <= F0) & (Fc + 3 > 1e-9)
ratio_w = np.where(inS, 0.5 * gc2 / np.maximum(Fc + 3, 1e-300), np.inf)
iw = np.unravel_index(np.argmin(ratio_w), ratio_w.shape)
mu_w = ratio_w[iw]
print(f"  the least ratio is attained at (beta, gamma) = ({Bc[iw]:.3f}, {Gc[iw]:.3f}), where F = {Fc[iw]:.4f}")
print(f"PL on the worked example: least (1/2)|grad F|^2 / (F - F*) on the sublevel set of F(0.1, -0.1) = {F0:.4f}: "
      f"mu = {mu_w:.4f}; at the minimiser the Hessian is diag(16, 4)")
th = np.array([beta0, gamma0]); gaps = [F_closed(*th) - Fstar]
for k in range(60):
    th = th - alpha * grad_closed(*th); gaps.append(F_closed(*th) - Fstar)
gaps = np.array(gaps)
with open(f"{OUT}/pl_gd.dat", "w") as fh:
    fh.write("k gap bound local\n")
    for k, gp in enumerate(gaps):
        fh.write(f"{k} {max(gp, 1e-16):.6e} {(1 - mu_w / Lb)**k * gaps[0]:.6e} {0.64**k * gaps[0]:.6e}\n")
assert np.all(gaps <= (1 - mu_w / Lb)**np.arange(len(gaps)) * gaps[0] + 1e-15)
print(f"  gradient descent, step 1/20: F - F* = {gaps[10]:.2e} after 10 steps, {gaps[30]:.2e} after 30, ratio of the last "
      f"two {gaps[60] / gaps[59]:.4f} (the slow direction: (1 - 4/20)^2 = 0.64); the bound (1 - mu/L)^k gives "
      f"{(1 - mu_w / Lb)**30 * gaps[0]:.2e} after 30")

# --------------------------------------------------------------------------- the constants of QAOA of depth p
def graph_data(n_, edges_):
    bits_ = np.array([[(x >> (n_ - 1 - q)) & 1 for q in range(n_)] for x in range(2**n_)])
    z_ = 1 - 2 * bits_
    Hn = np.array([[1.0]])
    for _ in range(n_):
        Hn = np.kron(Hn, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    cut_ = np.array([sum(b[i] != b[j] for i, j in edges_) for b in bits_]).astype(float)
    return dict(n=n_, m=len(edges_), cut=cut_, maxcut=cut_.max(), Hn=Hn, xs=z_.sum(1).astype(float),
                zzh=np.sum([z_[:, i] * z_[:, j] for i, j in edges_], axis=0) / 2)
def qaoa_mixer(psi, beta, g):                   # exp(-i beta sum_i X_i), that is R_X(2 beta) on every qubit
    return ((psi @ g["Hn"]) * np.exp(-1j * beta[:, None] * g["xs"])) @ g["Hn"]
def qaoa_forward(th, g):                        # th of shape (R, 2p): beta_1..beta_p, gamma_1..gamma_p
    p_ = th.shape[1] // 2
    psi = np.full((th.shape[0], 2**g["n"]), 2**(-g["n"] / 2), complex)
    states = []
    for k in range(p_):
        psi = psi * np.exp(-1j * th[:, p_ + k][:, None] * g["zzh"]); states.append(psi)
        psi = qaoa_mixer(psi, th[:, k], g); states.append(psi)
    return psi, states
def qaoa_cost(th, g): return np.abs(qaoa_forward(th, g)[0])**2 @ (-g["cut"])
def qaoa_grad(th, g):
    """The exact gradient by adjoint differentiation: dC/dt = 2 Im <lambda|G|psi> after each gate exp(-i t G)."""
    p_ = th.shape[1] // 2
    psi, states = qaoa_forward(th, g)
    lam, out = psi * (-g["cut"]), np.zeros_like(th)
    for k in range(p_ - 1, -1, -1):
        Gs = ((states[2 * k + 1] @ g["Hn"]) * g["xs"]) @ g["Hn"]
        out[:, k] = 2 * np.imag(np.sum(lam.conj() * Gs, 1))
        lam = qaoa_mixer(lam, -th[:, k], g)
        out[:, p_ + k] = 2 * np.imag(np.sum(lam.conj() * (states[2 * k] * g["zzh"]), 1))
        lam = lam * np.exp(1j * th[:, p_ + k][:, None] * g["zzh"])
    return out
def qaoa_hess(th, g, h=1e-5):
    P_ = th.shape[1]
    H_ = np.zeros((th.shape[0], P_, P_))
    for j in range(P_):
        e_ = np.zeros(P_); e_[j] = h
        H_[:, j, :] = (qaoa_grad(th + e_, g) - qaoa_grad(th - e_, g)) / (2 * h)
    return (H_ + np.transpose(H_, (0, 2, 1))) / 2
cycle4 = graph_data(4, edges)
cubic8 = graph_data(8, [tuple(map(int, l.split())) for l in open("figures/ch04/maxcut_graph.dat").read().split("\n")[1:] if l.strip()])
assert abs(qaoa_cost(np.array([[0.2, -0.3]]), cycle4)[0] - F_closed(0.2, -0.3)) < 1e-12
for g_ in (cycle4, cubic8):
    t_ = rng.uniform(-np.pi, np.pi, (3, 6))
    num = np.array([(qaoa_cost(t_ + 1e-6 * e_, g_) - qaoa_cost(t_ - 1e-6 * e_, g_)) / 2e-6 for e_ in np.eye(6)]).T
    assert np.allclose(num, qaoa_grad(t_, g_), atol=1e-6)
def largest(fun, P_, top=10, rounds=15, per=6):
    """The largest value of fun found by a local search: 2000 random points, then the 10 best perturbed
    15 times, 6 candidates each, with a spread decreasing from 0.2 to 0.02; a lower bound on the maximum."""
    t_ = rng.uniform(-np.pi, np.pi, (2000, P_))
    v_ = fun(t_)
    best, bv = t_[np.argsort(v_)[-top:]], np.sort(v_)[-top:]
    for r_ in range(rounds):
        cand = np.repeat(best, per, 0) + 0.2 * 0.1**(r_ / (rounds - 1)) * rng.normal(size=(top * per, P_))
        cv = fun(cand).reshape(top, per)
        j_ = cv.argmax(1); up = cv[np.arange(top), j_] > bv
        best[up] = cand.reshape(top, per, -1)[np.arange(top), j_][up]; bv[up] = cv[np.arange(top), j_][up]
    return bv.max()
rows = []
for p_ in range(1, 7):
    row = [p_]
    for g_ in (cycle4, cubic8):
        R2, W_ = p_ * (g_["maxcut"]**2 + 4 * g_["n"]**2), g_["maxcut"] / 2
        gmax_ = largest(lambda t_: np.linalg.norm(qaoa_grad(t_, g_), axis=1), 2 * p_)
        hmax_ = largest(lambda t_: np.abs(np.linalg.eigvalsh(qaoa_hess(t_, g_))).max(axis=1), 2 * p_)
        assert gmax_ <= np.sqrt(R2) * W_ and hmax_ <= R2 * W_
        row += [gmax_, np.sqrt(R2) * W_, hmax_, R2 * W_]
    rows.append(row)
    print(f"QAOA depth {p_}: 4-cycle max|grad| {row[1]:.2f} (bound {row[2]:.1f}), max|Hessian| {row[3]:.2f} (bound {row[4]:.0f}); "
          f"3-regular max|grad| {row[5]:.2f} (bound {row[6]:.1f}), max|Hessian| {row[7]:.2f} (bound {row[8]:.0f})")
with open(f"{OUT}/qaoa_constants.dat", "w") as fh:
    fh.write("p cgrad cgradb chess chessb rgrad rgradb rhess rhessb\n")
    for row in rows:
        fh.write(" ".join(f"{v:.6g}" for v in row) + "\n")
print(f"  4-cycle: maxcut {cycle4['maxcut']:.0f}, n = 4, m = 4; 3-regular graph: maxcut {cubic8['maxcut']:.0f}, n = 8, m = 12; "
      f"circuits for a gradient gate by gate 2p(m + n): {2 * (4 + 4)}p and {2 * (12 + 8)}p")

# --------------------------------------------------------------------------- two costs that a device cannot distinguish
dl2 = 0.6
b2 = 2 * np.sin(dl2 / 2)
tt = np.linspace(-np.pi, np.pi, 241)
assert np.max(np.abs(-np.cos(tt) + np.cos(tt - dl2))) <= b2 + 1e-12
tf = np.linspace(-np.pi, np.pi, 200001)
mx = np.maximum(np.abs(np.sin(tf)), np.abs(np.sin(tf - dl2)))
assert mx.min() >= np.sin(dl2) / 2
with open(f"{OUT}/twocost_band.dat", "w") as fh:
    fh.write("theta c\n")
    for t_ in tt:
        fh.write(f"{t_:.5f} {-np.cos(t_) + b2:.5f}\n")
    for t_ in tt[::-1]:
        fh.write(f"{t_:.5f} {-np.cos(t_) - b2:.5f}\n")
print(f"two costs, delta = {dl2}: b = {b2:.4f}; the larger slope is at least {mx.min():.4f} = sin(delta/2) = {np.sin(dl2 / 2):.4f}, "
      f"against (1/2) sin delta = {np.sin(dl2) / 2:.4f} in the proof")

# --------------------------------------------------------------------------- the three terms of the bound against the step
M_s, K_s = 100, 1000
alphas = np.exp(np.linspace(np.log(0.002), np.log(0.1), 13))
with open(f"{OUT}/sgd_steps.dat", "w") as fh:
    fh.write("alpha avg progress noise bound\n")
    paths_s = {}
    for i_, a_ in enumerate(alphas):
        th = np.tile([beta0, gamma0], (runs, 1)).astype(float); acc = 0.0
        path_ = [th[0].copy()]
        for k in range(K_s):
            acc += np.mean(np.sum(grad_closed(th[:, 0], th[:, 1])**2, axis=0))
            th -= a_ * psr_fast(th[:, 0], th[:, 1], M_s)
            path_.append(th[0].copy())
        paths_s[i_] = np.array(path_)
        pr_, no_ = 2 * (F0 - Fstar) / (a_ * K_s), a_ * Lb * MV / M_s
        fh.write(f"{a_:.6e} {acc / K_s:.6e} {pr_:.6e} {no_:.6e} {pr_ + no_:.6e}\n")
        print(f"SGD with alpha = {a_:.4f}, M = {M_s}, K = {K_s}: average {acc / K_s:.4f}, bound {pr_ + no_:.4f}" + ("" if a_ <= 1 / Lb else " (beyond 1/L)"))
a_best = np.sqrt(2 * (F0 - Fstar) / (Lb * MV / M_s * K_s))
# the path of the first run for the smallest step, the step nearest the best, and the largest
with open(f"{OUT}/sgd_steps_paths.dat", "w") as fh:
    fh.write("k " + " ".join(f"b{i_} g{i_}" for i_ in (0, 8, 12)) + "\n")
    for k in range(K_s + 1):
        fh.write(f"{k} " + " ".join(f"{paths_s[i_][k, 0]:.6f} {paths_s[i_][k, 1]:.6f}" for i_ in (0, 8, 12)) + "\n")
for i_ in (0, 8, 12):
    pth = paths_s[i_]; g2p = np.sum(grad_closed(pth[:, 0], pth[:, 1])**2, axis=0)
    dist = np.hypot(pth[:, 0] - np.pi / 8, pth[:, 1] + np.pi / 4)
    near = np.nonzero(dist < 0.05)[0]
    print(f"  the first run with alpha = {alphas[i_]:.4f}: within 0.05 of the minimiser first at k = "
          f"{near[0] if len(near) else 'never'}; distance after {K_s} steps {dist[-1]:.4f}; root-mean-square distance over the "
          f"last 500 steps {np.sqrt(np.mean(dist[-500:]**2)):.4f}; average |grad F|^2 {g2p[:-1].mean():.4f}")
print(f"  the best step of the bound {a_best:.4f}, where it is {2 * np.sqrt(2 * (F0 - Fstar) * Lb * MV / M_s / K_s):.4f}")

# --------------------------------------------------------------------------- decreasing steps
# |grad F(theta_k)|^2, the mean over the runs, for the shift rule with a constant step and with the gains
# a_k = a / (k + 1 + A)^0.602, a_0 = 0.05, and for simultaneous perturbation with the gains a_k, a_0 = 0.01, and
# c_k = 0.2 / (k + 1)^0.101; M = 100 copies of each circuit
K_d, M_d, A_d, s_d, t_d, c_d = 10000, 100, 100, 0.602, 0.101, 0.2
a_d = alpha * (1 + A_d)**s_d                     # so that a_0 = 0.05, the constant step
a_s = 0.01 * (1 + A_d)**s_d                      # a_0 = 0.01 for simultaneous perturbation
kd = np.unique(np.round(np.exp(np.linspace(0, np.log(K_d), 60))).astype(int)) - 1
def run_decr(kind, dl):
    th = np.tile([beta0, gamma0], (runs, 1)).astype(float)
    out = np.zeros(K_d); snaps = {}
    for k in range(K_d):
        out[k] = np.mean(np.sum(grad_closed(th[:, 0], th[:, 1])**2, axis=0))
        a_k = alpha if kind == "const" else (a_s if kind == "spsa" else a_d) / (k + 1 + A_d)**s_d
        if kind == "spsa":
            th -= a_k * spsa_grad(th, c_d / (k + 1)**t_d, M_d, dl)
        else:
            th -= a_k * psr_fast(th[:, 0], th[:, 1], M_d, dl)
        if k + 1 in (100, 1000, K_d):
            snaps[k + 1] = th.copy()
    return out, snaps
for name, dl in (("unbiased", 0.0), ("biased", delta)):
    runs_d = {kind: run_decr(kind, dl) for kind in ("const", "decr", "spsa")}
    res = {kind: r_[0] for kind, r_ in runs_d.items()}
    snaps_d = {kind: r_[1] for kind, r_ in runs_d.items()}
    cols_d = [("const", K_d), ("decr", 100), ("decr", 1000), ("decr", K_d), ("spsa", K_d)]
    with open(f"{OUT}/decr_cloud_{name}.dat", "w") as fh:
        fh.write(" ".join(f"b{kind}{k_} g{kind}{k_}" for kind, k_ in cols_d) + "\n")
        for r in range(runs):
            fh.write(" ".join(f"{snaps_d[kind][k_][r, 0]:.6f} {snaps_d[kind][k_][r, 1]:.6f}" for kind, k_ in cols_d) + "\n")
    mn_ = np.array([np.pi / 8 - (dl / 2), -np.pi / 4])
    print(f"decreasing steps, {name}: root-mean-square distance of the iterates from the minimiser of the device's cost "
          + "; ".join(f"{kind} at k = {k_} {np.sqrt(np.mean(np.sum((snaps_d[kind][k_] - mn_)**2, 1))):.4f}" for kind, k_ in cols_d)
          + f"; mean distance of the centre of the decreasing-step cloud at k = {K_d} from the minimiser of F "
          f"{np.hypot(*(snaps_d['decr'][K_d].mean(0) - [np.pi / 8, -np.pi / 4])):.4f}")
    with open(f"{OUT}/decr_{name}.dat", "w") as fh:
        fh.write("k const decr spsa\n")
        for k in kd:
            fh.write(f"{k + 1} " + " ".join(f"{res[kind][k]:.6e}" for kind in ("const", "decr", "spsa")) + "\n")
    print(f"decreasing steps, {name}, M = {M_d}: mean |grad F|^2 at k = 100, 1000, 10000: "
          + "; ".join(f"{kind} " + ", ".join(f"{res[kind][k - 1]:.4f}" for k in (100, 1000, 10000)) for kind in res))
print(f"  gains a_k = {a_d:.4f} / (k + {1 + A_d})^{s_d}: a_0 = {a_d / (1 + A_d)**s_d:.4f}, a_10000 = {a_d / (K_d + A_d)**s_d:.5f}; "
      f"for simultaneous perturbation a_0 = {a_s / (1 + A_d)**s_d:.4f}, c_k = {c_d} / (k + 1)^{t_d}, c_10000 = {c_d / K_d**t_d:.4f}; "
      f"s + 4t = {s_d + 4 * t_d:.3f}, 2s - 2t = {2 * s_d - 2 * t_d:.3f}")

# --------------------------------------------------------------------------- the derivative-free estimators on the worked example
# running averages of E|grad F(theta_k)|^2 against the circuits: the shift rule (16 circuits, alpha = 1/20), simultaneous
# perturbation (2 circuits, c = 0.1) with the step 1/((8d + 1) L) of its proposition and with 1/20, and the two-point
# estimator (2 circuits, c = 0.1) with 1/((16d + 65) L); M = 100 copies of each circuit
M_f, N_f, c_f, d_f = 100, 48000, 0.1, 2
L3_f = Lb**1.5                                   # |D^3 F| <= |R|^3 W = 20^(3/2) by the lemma on derivatives
kap_f = L3_f * d_f**1.5 / 6
methods = {"psr": (alpha, 16), "spsa": (1 / ((8 * d_f + 1) * Lb), 2), "spsafast": (alpha, 2), "twopoint": (1 / ((16 * d_f + 65) * Lb), 2)}
nf = np.unique(np.round(np.exp(np.linspace(np.log(16), np.log(N_f), 60))).astype(int))
def run_dfo(kind, dl):
    a_, per = methods[kind]
    K_ = N_f // per
    th = np.tile([beta0, gamma0], (runs, 1)).astype(float)
    g2 = np.zeros(K_); path_ = [th[0].copy()]
    for k in range(K_):
        g2[k] = np.mean(np.sum(grad_closed(th[:, 0], th[:, 1])**2, axis=0))
        if kind == "psr":
            th -= a_ * psr_fast(th[:, 0], th[:, 1], M_f, dl)
        elif kind == "twopoint":
            th -= a_ * twopoint_grad(th, c_f, M_f, dl)
        else:
            th -= a_ * spsa_grad(th, c_f, M_f, dl)
        path_.append(th[0].copy())
    return np.cumsum(g2) / np.arange(1, K_ + 1), np.array(path_), th
sig2_f = 4.0                                     # values in [-4, 0]
for name, dl in (("unbiased", 0.0), ("biased", delta)):
    b_f = 2 * np.sin(dl)                         # the bias of a value on the angles of QAOA
    runs_f = {kind: run_dfo(kind, dl) for kind in methods}
    res = {kind: r_[0] for kind, r_ in runs_f.items()}
    # the first run after every 16 circuits, up to 8000, and all the runs after N_f circuits
    with open(f"{OUT}/dfo_paths_{name}.dat", "w") as fh:
        fh.write("circuits " + " ".join(f"b{kind} g{kind}" for kind in methods) + "\n")
        for nc in range(0, 8001, 16):
            fh.write(f"{nc} " + " ".join(f"{runs_f[kind][1][nc // methods[kind][1], 0]:.6f} "
                                         f"{runs_f[kind][1][nc // methods[kind][1], 1]:.6f}" for kind in methods) + "\n")
    with open(f"{OUT}/dfo_cloud_{name}.dat", "w") as fh:
        fh.write(" ".join(f"b{kind} g{kind}" for kind in methods) + "\n")
        for r in range(runs):
            fh.write(" ".join(f"{runs_f[kind][2][r, 0]:.6f} {runs_f[kind][2][r, 1]:.6f}" for kind in methods) + "\n")
    mn_ = np.array([np.pi / 8 - (dl / 2), -np.pi / 4])
    for kind in methods:
        pth = runs_f[kind][1]; per = methods[kind][1]
        dist = np.hypot(pth[:, 0] - mn_[0], pth[:, 1] - mn_[1])
        near = np.nonzero(dist < 0.05)[0]
        cl = runs_f[kind][2]
        print(f"  {name}, {kind}: the first run within 0.05 of the minimiser of the device's cost after "
              f"{per * near[0] if len(near) else 'never'} circuits; after {N_f} circuits the {runs} runs lie at a "
              f"root-mean-square distance {np.sqrt(np.mean(np.sum((cl - mn_)**2, 1))):.4f} from it, centre "
              f"({cl[:, 0].mean():.4f}, {cl[:, 1].mean():.4f})")
    with open(f"{OUT}/dfo_{name}.dat", "w") as fh:
        fh.write("circuits " + " ".join(methods) + " bndpsr bndspsa\n")
        for nc in nf:
            vals_ = [res[kind][max(nc // methods[kind][1], 1) - 1] for kind in methods]
            Kp, Ks = max(nc // 16, 1), max(nc // 2, 1)
            bp = 2 * (F0 - Fstar) / (alpha * Kp) + alpha * Lb * MV / M_f + (zeta**2 if dl else 0)
            a_sp = methods["spsa"][0]
            bs_ = (2 * (F0 - Fstar) / (a_sp * Ks) + a_sp * Lb * (Lb**2 * c_f**2 * d_f**3 + d_f * (4 * b_f**2 + sig2_f / M_f) / (2 * c_f**2))
                   + 2 * (kap_f * c_f**2 + b_f / c_f)**2)
            fh.write(f"{nc} " + " ".join(f"{v:.6e}" for v in vals_) + f" {bp:.6e} {bs_:.6e}\n")
    print(f"derivative-free, {name}, M = {M_f}, after {N_f} circuits: " + "; ".join(f"{kind} {res[kind][-1]:.4f}" for kind in methods)
          + f"; floor of the proposition for SPSA {2 * (kap_f * c_f**2 + b_f / c_f)**2:.3f}")
    for kind in methods:
        r_ = res[kind]; per = methods[kind][1]
        hit = np.nonzero(r_ < 0.5)[0]
        print(f"  {kind}: running average below 0.5 after {per * (hit[0] + 1) if len(hit) else 'never'} circuits")
print(f"  steps: shift rule {alpha}, simultaneous perturbation {methods['spsa'][0]:.5f}, two-point {methods['twopoint'][0]:.6f}; "
      f"kappa = {kap_f:.2f}; the two-point floor (32/pi) b^2/c^2 + (c^2/2) L^2 (d + 3)^3 at c = 0.1 is "
      f"{(32 / np.pi) * (2 * np.sin(delta))**2 / c_f**2 + c_f**2 / 2 * Lb**2 * (d_f + 3)**3:.1f}, above max |grad F|^2 = 16")

# =========================================================================== pictures before the lemmas
rng = np.random.default_rng(2034)       # again a generator of its own, so that the numbers above do not change

# --------------------------------------------------------------------------- the descent lemma in the plane
# near the start, the paraboloid F(theta_0) + grad F^T h + (L/2)|h|^2 with L = 20 is at most F(theta_0) on the disc
# of radius |grad F(theta_0)| / L about theta_1 = theta_0 - grad F(theta_0) / L, and F is below the paraboloid
t0 = np.array([beta0, gamma0]); g0 = grad_closed(*t0)
r0 = np.hypot(*g0) / Lb; t1 = t0 - g0 / Lb
rr, aa = np.meshgrid(np.linspace(0, r0, 201), np.linspace(0, 2 * np.pi, 721))
Fdisc = F_closed(t1[0] + rr * np.cos(aa), t1[1] + rr * np.sin(aa))
Pdisc = F0 - np.sum(g0**2) / (2 * Lb) + Lb / 2 * rr**2
assert np.all(Fdisc <= Pdisc + 1e-12) and np.all(Pdisc <= F0 + 1e-12)
print(f"descent lemma in the plane: grad F(theta_0) = ({g0[0]:.4f}, {g0[1]:.4f}), norm {np.hypot(*g0):.4f}; the disc of "
      f"radius |grad F|/L = {r0:.4f} about theta_1 = ({t1[0]:.4f}, {t1[1]:.4f}), on which F is at most {Fdisc.max():.5f} "
      f"<= F(theta_0) = {F0:.5f}; guaranteed decrease {np.sum(g0**2) / (2 * Lb):.4f}, F(theta_1) = {F_closed(*t1):.4f}, "
      f"actual decrease {F0 - F_closed(*t1):.4f}")
lev_d = [-2.02, -2.05, F0, -2.11, -2.15, -2.2, -2.26, -2.33]
with open(f"{OUT}/descent_plane.dat", "w") as fh:
    fh.write("beta " + " ".join(f"l{i}" for i in range(len(lev_d))) + "\n")
    for b_ in np.linspace(0.005, 0.26, 511):
        q_ = (np.array(lev_d) + 2) / np.sin(4 * b_)
        fh.write(f"{b_:.5f} " + " ".join(f"{0.5 * np.arcsin(x):.6f}" if abs(x) <= 1 else "nan" for x in q_) + "\n")

# --------------------------------------------------------------------------- gradient descent on the landscape
# in the cell 0 < beta < pi/4, -pi/2 < gamma < 0, with a = 4 beta - pi/2 and b = 2 gamma + pi/2, F = -2 - cos a cos b;
# the level curve F = -2 - q is cos a cos b = q, traced by a = +-arccos(q^w), b = +-arccos(q^(1-w)), 0 <= w <= 1
def level_curve(q, N=200):
    w = (1 - np.cos(np.pi * np.linspace(0, 1, N))) / 2
    a_, b_ = np.arccos(q**w), np.arccos(q**(1 - w))
    A_ = np.r_[a_, a_[::-1], -a_, -a_[::-1]]
    B_ = np.r_[b_, -b_[::-1], -b_, b_[::-1]]
    return (A_ + np.pi / 2) / 4, (B_ - np.pi / 2) / 2
qs = [-(F0 + 2), 0.25, 0.5, 0.75, 0.9, 0.99, 0.97, 0.94]
curves = [level_curve(q) for q in qs]
for (cb, cg), q in zip(curves, qs):
    assert np.allclose(F_closed(cb, cg), -2 - q)
with open(f"{OUT}/gd_levels.dat", "w") as fh:
    fh.write(" ".join(f"b{i} g{i}" for i in range(len(qs))) + "\n")
    for j in range(len(curves[0][0])):
        fh.write(" ".join(f"{cb[j]:.6f} {cg[j]:.6f}" for cb, cg in curves) + "\n")
th = t0.copy(); path = [th.copy()]
for k in range(40):
    th = th - grad_closed(*th) / Lb; path.append(th.copy())
path = np.array(path)
Fp = F_closed(path[:, 0], path[:, 1]); G2 = np.sum(grad_closed(path[:, 0], path[:, 1])**2, axis=0)
assert np.all(np.diff(Fp) <= -G2[:-1] / (2 * Lb) + 1e-15)
with open(f"{OUT}/gd_path.dat", "w") as fh:
    fh.write("k beta gamma F grad2 guar dbeta dgamma\n")
    for k in range(len(path)):
        st_ = -grad_closed(*path[k]) / Lb
        fh.write(f"{k} {path[k, 0]:.6f} {path[k, 1]:.6f} {Fp[k]:.6f} {G2[k]:.6e} {Fp[k] - G2[k] / (2 * Lb):.6f} "
                 f"{st_[0]:.6f} {st_[1]:.6f}\n")
print(f"gradient descent on the landscape, step 1/L = 1/20: F(theta_0) - F* = {F0 - Fstar:.4f}, 2 L (F(theta_0) - F*) = "
      f"{2 * Lb * (F0 - Fstar):.3f}; |grad F(theta_k)|^2 for k = 0..5: {', '.join(f'{x:.3f}' for x in G2[:6])}")
for K_ in (1, 5, 10):
    print(f"  K = {K_}: min_k |grad F|^2 = {G2[:K_].min():.4f}, average {G2[:K_].mean():.4f}, the bound "
          f"{2 * Lb * (F0 - Fstar) / K_:.3f}; F(theta_K) - F* = {Fp[K_] - Fstar:.4f}; the guaranteed decreases add up to "
          f"{np.sum(G2[:K_]) / (2 * Lb):.4f}, the actual ones to {F0 - Fp[K_]:.4f}")
print(f"  decreases of the first steps: guaranteed {', '.join(f'{x:.4f}' for x in G2[:4] / (2 * Lb))}; actual "
      f"{', '.join(f'{x:.4f}' for x in -np.diff(Fp)[:4])}; all {len(path) - 1} guaranteed decreases add up to "
      f"{np.sum(G2[:-1]) / (2 * Lb):.4f} <= {F0 - Fstar:.4f}")
# the curvature: the Hessian at the minimiser is diag(16, 4), and the level curves there are ellipses with the axes
# in the ratio 2 : 1, the shorter along beta
print(f"  the Hessian at the minimiser: diag(16, 4); the largest |Hessian| on the plane {lam.max():.4f} <= L = {Lb:.0f}")

# --------------------------------------------------------------------------- the ratio of the Polyak-Lojasiewicz inequality
# (1/2)|grad F|^2 / (F - F*) on the cell, on a grid that misses the minimiser; its logarithm for the colours
Br, Gr = np.meshgrid(np.linspace(0, np.pi / 4, 32), np.linspace(-np.pi / 2, 0, 63))
rat = 0.5 * np.sum(grad_closed(Br, Gr)**2, axis=0) / (F_closed(Br, Gr) - Fstar)
with open(f"{OUT}/pl_ratio.dat", "w") as fh:
    fh.write("beta gamma lr\n")
    for j in range(Br.shape[0]):
        for i in range(Br.shape[1]):
            fh.write(f"{Br[j, i]:.5f} {Gr[j, i]:.5f} {np.log10(max(rat[j, i], 1e-2)):.4f}\n")
inS_ = F_closed(Br, Gr) <= F0
nearm = np.hypot(Br - np.pi / 8, Gr + np.pi / 4) < 0.05
print(f"the PL ratio on the cell: at most {rat.max():.3f}; on the sublevel set of theta_0 at least {rat[inS_].min():.4f} "
      f"(mu = {mu_w:.4f} on the finer grid); within 0.05 of the minimiser between {rat[nearm].min():.3f} and "
      f"{rat[nearm].max():.3f}; at the corners {rat[0, 0]:.1e}; on the edges 8 sin^2 2 gamma, at most 8")
assert rat[inS_].min() >= mu_w - 1e-9 and 4 - 0.1 < rat[nearm].min() and rat[nearm].max() < 16 + 0.1

# --------------------------------------------------------------------------- estimates of the gradient
# at theta_0, on the device whose mixer is over-rotated by delta: 200 estimates by the shift rule with M = 1000 copies
# of each circuit, and the two estimates of simultaneous perturbation with c = 0.1 and exact values
Mn, nn, cn = 1000, 200, 0.1
gdev = grad_closed(beta0 + delta / 2, gamma0)            # the mean of the shift rule on the device
est = psr_fast(np.full(nn, beta0), np.full(nn, gamma0), Mn, delta)
Vn = var_sum(np.array([beta0]), np.array([gamma0]), delta)[0] / Mn
with open(f"{OUT}/noise_estimates.dat", "w") as fh:
    fh.write("gb gg\n")
    for e_ in est:
        fh.write(f"{e_[0]:.5f} {e_[1]:.5f}\n")
Ft = lambda th_: F_closed(th_[0] + delta / 2, th_[1])
sp = [(Ft(t0 + cn * D) - Ft(t0 - cn * D)) / (2 * cn) * D for D in (np.array([1.0, 1.0]), np.array([1.0, -1.0]))]
spm = (sp[0] + sp[1]) / 2
bias_n = gdev - g0
print(f"estimates at theta_0 on the over-rotated device: grad F = ({g0[0]:.4f}, {g0[1]:.4f}); the mean of the shift rule "
      f"({gdev[0]:.4f}, {gdev[1]:.4f}); the bias ({bias_n[0]:.4f}, {bias_n[1]:.4f}), of norm {np.hypot(*bias_n):.4f} "
      f"<= zeta = {zeta:.4f}")
print(f"  shift rule, M = {Mn}: V(theta_0) = {Vn:.5f}, sqrt V = {np.sqrt(Vn):.4f}; the {nn} estimates have the mean "
      f"({est[:, 0].mean():.4f}, {est[:, 1].mean():.4f}) and the root-mean-square distance from it "
      f"{np.sqrt(np.mean(np.sum((est - est.mean(0))**2, 1))):.4f}; {np.mean(np.hypot(*(est - gdev).T) <= np.sqrt(Vn)):.2f} "
      f"of them within sqrt V of the mean")
print(f"  simultaneous perturbation, c = {cn}, exact values: ({sp[0][0]:.4f}, {sp[0][1]:.4f}) for Delta = +-(1, 1), "
      f"({sp[1][0]:.4f}, {sp[1][1]:.4f}) for +-(1, -1); the mean ({spm[0]:.4f}, {spm[1]:.4f}), "
      f"{np.hypot(*(spm - gdev)):.4f} from that of the shift rule; E|rho|^2 = {np.sum((sp[0] - spm)**2):.4f} against "
      f"|mean|^2 = {np.sum(spm**2):.4f}")

# --------------------------------------------------------------------------- secants on a biased cost
# the cost -cos(phi) and the values of a device, -cos(phi) + b sin(25 phi + 1), b = 0.06, at phi_0 = 1: the central
# difference of half-width c has the error of the device at most b/c and that of the Taylor expansion about kappa c^2,
# kappa = 1/6; the shift rule, half the difference at phi_0 +- pi/2, is exact for -cos and has the error at most b
b_s, phi_s = 0.06, 1.0
Cs = lambda x: -np.cos(x)
Ct = lambda x: -np.cos(x) + b_s * np.sin(25 * x + 1)
c_best = (b_s / (2 / 6))**(1 / 3)
print(f"secants: C = -cos, b = {b_s}, phi_0 = {phi_s}: C'(phi_0) = {np.sin(phi_s):.4f}; best c = (b / 2 kappa)^(1/3) = {c_best:.4f}")
for c_ in (0.06, c_best, 1.3):
    sl_t, sl_c = (Ct(phi_s + c_) - Ct(phi_s - c_)) / (2 * c_), (Cs(phi_s + c_) - Cs(phi_s - c_)) / (2 * c_)
    print(f"  c = {c_:.4f}: the points ({phi_s - c_:.4f}, {Ct(phi_s - c_):.4f}) and ({phi_s + c_:.4f}, {Ct(phi_s + c_):.4f}); "
          f"the slope of the secant of the device {sl_t:.4f}, error {sl_t - np.sin(phi_s):+.4f}; of the cost {sl_c:.4f}, "
          f"error {sl_c - np.sin(phi_s):+.4f}; the bounds b/c = {b_s / c_:.4f} and kappa c^2 = {c_**2 / 6:.4f}")
ps_ = (Ct(phi_s + np.pi / 2) - Ct(phi_s - np.pi / 2)) / 2
print(f"  the shift rule: the points ({phi_s - np.pi / 2:.4f}, {Ct(phi_s - np.pi / 2):.4f}) and ({phi_s + np.pi / 2:.4f}, "
      f"{Ct(phi_s + np.pi / 2):.4f}); the estimate {ps_:.4f}, error {ps_ - np.sin(phi_s):+.4f} (at most b = {b_s}); the slope "
      f"of the secant through them {(Ct(phi_s + np.pi / 2) - Ct(phi_s - np.pi / 2)) / np.pi:.4f}, times pi/2")

# =========================================================================== the model of the noise; the estimators on the landscape
rng = np.random.default_rng(2036)       # again a generator of its own, so that the numbers above do not change

# --------------------------------------------------------------------------- the model of the noise
# along the line gamma = -pi/4 through the minimiser, on the device whose mixer is over-rotated by delta: the cost C, the
# cost of the device C~ = E C^ = C(beta + delta/2), the band C -+ b with b = 2 sin delta, one estimate with M = 1000
# copies at each of 33 angles, and at beta = 0.1 the standard deviation of an estimate for M = 10, 100, 1000
gm, bm, Mm = -np.pi / 4, 2 * np.sin(delta), 1000
Bm = np.linspace(0, np.pi / 4, 401)
Cm, Ctm = F_closed(Bm, gm), F_closed(Bm + delta / 2, gm)
assert np.allclose(probs(*angles(Bm, np.full_like(Bm, gm), delta)) @ vals, Ctm) and np.abs(Ctm - Cm).max() <= bm + 1e-12
# the variance of one measured value is at most sigma^2 = 2, on the ideal device and on this one
Bv, Gv = np.meshgrid(np.linspace(0, np.pi, 161), np.linspace(-np.pi, np.pi, 321))
s2v = [(lambda p_: p_ @ vals**2 - (p_ @ vals)**2)(probs(*angles(Bv.ravel(), Gv.ravel(), dl_))) for dl_ in (0.0, delta)]
sig2 = 2.0
assert max(s2v[0].max(), s2v[1].max()) <= sig2 + 1e-9
with open(f"{OUT}/noise_model.dat", "w") as fh:
    fh.write("beta C Ct lo hi\n")
    for j in range(len(Bm)):
        fh.write(f"{Bm[j]:.5f} {Cm[j]:.5f} {Ctm[j]:.5f} {Cm[j] - bm:.5f} {Cm[j] + bm:.5f}\n")
Be = np.linspace(0, np.pi / 4, 33)
Ee = F_est(Be, np.full_like(Be, gm), Mm, delta)
with open(f"{OUT}/noise_model_est.dat", "w") as fh:
    fh.write("beta est\n")
    for b_, e_ in zip(Be, Ee):
        fh.write(f"{b_:.5f} {e_:.5f}\n")
bq = 0.1
pq = probs(*angles(np.array([bq]), np.array([gm]), delta))[0]
mq, sq = pq @ vals, np.sqrt(pq @ vals**2 - (pq @ vals)**2)
with open(f"{OUT}/noise_model_bars.dat", "w") as fh:
    fh.write("M beta mean sd C\n")
    for M_ in (10, 100, 1000):
        fh.write(f"{M_} {bq} {mq:.5f} {sq / np.sqrt(M_):.5f} {F_closed(bq, gm):.5f}\n")
print(f"the model of the noise along gamma = -pi/4, over-rotated mixer, delta = {delta}: |C~ - C| at most "
      f"{np.abs(Ctm - Cm).max():.4f} = b = 2 sin delta = {bm:.4f}, C~ - C = {Ctm[0] - Cm[0]:+.4f} at beta = 0 and "
      f"{Ctm[-1] - Cm[-1]:+.4f} at pi/4; the minimiser of C~ at pi/8 - delta/2 = {np.pi / 8 - delta / 2:.4f}; the variance of "
      f"one value at most {max(s2v[0].max(), s2v[1].max()):.4f} (sigma^2 = {sig2:.0f}); sigma^2 / b^2 = {sig2 / bm**2:.0f}")
bs_ = (np.pi - delta) / 4
print(f"  the bias attains b at beta = (pi - delta)/4 = {bs_:.4f}: C = {F_closed(bs_, gm):.4f}, C~ = "
      f"{F_closed(bs_ + delta / 2, gm):.4f}")
print(f"  at beta = {bq}: C = {F_closed(bq, gm):.4f}, C~ = {mq:.4f}, C~ - C = {mq - F_closed(bq, gm):+.4f}; the standard "
      f"deviation of one value {sq:.4f}, of an estimate {', '.join(f'{sq / np.sqrt(M_):.4f}' for M_ in (10, 100, 1000))} "
      f"for M = 10, 100, 1000; they equal the bias at M = {sq**2 / (mq - F_closed(bq, gm))**2:.0f}")
ide = F_closed(Be, gm); dev_ = F_closed(Be + delta / 2, gm)
print(f"  {len(Be)} estimates with M = {Mm}: root-mean-square distance {np.sqrt(np.mean((Ee - dev_)**2)):.4f} from C~ and "
      f"{np.sqrt(np.mean((Ee - ide)**2)):.4f} from C; within beta < 0.15 or beta > 0.64, "
      f"{np.mean(np.abs(Ee - dev_)[(Be < 0.15) | (Be > 0.64)] < np.abs(Ee - ide)[(Be < 0.15) | (Be > 0.64)]):.2f} of them "
      f"closer to C~ than to C")

# --------------------------------------------------------------------------- the estimators on the landscape
# exact values and c = 0.1, from theta_0: simultaneous perturbation with the step 1/(dL) = 1/40 and the two-point
# estimator with 1/((d+2)L) = 1/80, the steps that maximise the decrease that the descent lemma guarantees in
# expectation, since E|g|^2 = d |grad F|^2 and (d+2) |grad F|^2 as c -> 0; 80 iterations of 2 circuits each, against the
# 10 iterations of gradient descent, whose exact gradient takes the 16 circuits of the shift rule; 1000 runs of each
Fx = lambda th_: F_closed(th_[:, 0], th_[:, 1])
def spsa_exact(th_, c_):
    D = rng.choice([-1.0, 1.0], size=th_.shape)
    return ((Fx(th_ + c_ * D) - Fx(th_ - c_ * D)) / (2 * c_))[:, None] * D
def twopoint_exact(th_, c_):
    u = rng.normal(size=th_.shape)
    return ((Fx(th_ + c_ * u) - Fx(th_)) / c_)[:, None] * u
ce, Re, Ke = 0.1, 1000, 400
tmin = np.array([np.pi / 8, -np.pi / 4])
m2 = [np.mean(np.sum(f_(np.repeat(t0[None], 20000, 0), ce)**2, 1)) / np.sum(g0**2) for f_ in (spsa_exact, twopoint_exact)]
print(f"estimators on the landscape, exact values, c = {ce}: E|g|^2 / |grad F|^2 at theta_0 = {m2[0]:.3f} for simultaneous "
      f"perturbation (d = 2) and {m2[1]:.3f} for the two-point estimator (d + 2 = 4), from 20000 draws")
paths_e = {}
for nm, f_, a_ in (("spsa", spsa_exact, 1 / 40), ("two", twopoint_exact, 1 / 80)):
    th_ = np.repeat(t0[None], Re, 0); first = np.full(Re, -1); P_ = [th_[0].copy()]; Fm = []
    for k in range(Ke):
        th_ = th_ - a_ * f_(th_, ce); P_.append(th_[0].copy()); Fm.append(Fx(th_).mean() - Fstar)
        first[(first < 0) & (np.hypot(*(th_ - tmin).T) < 0.05)] = k + 1
    paths_e[nm] = np.array(P_)
    print(f"  {nm}, step 1/{1 / a_:.0f}: the first run within 0.05 of the minimiser after {first[0]} iterations, "
          f"{2 * first[0]} circuits; the median over {Re} runs {np.median(np.where(first < 0, 10**6, first)):.0f} iterations; "
          f"E F - F* after 8, 40, 80, 400 iterations: {Fm[7]:.4f}, {Fm[39]:.4f}, {Fm[79]:.4f}, {Fm[-1]:.4f}; "
          f"F - F* of the first run after 80: {F_closed(*P_[80]) - Fstar:.4f}")
kgd = np.argmax(np.hypot(*(path - tmin).T) < 0.05)
print(f"  gradient descent: within 0.05 after {kgd} iterations, {16 * kgd} circuits; F - F* after 1, 5, 10 iterations "
      f"(16, 80, 160 circuits): {Fp[1] - Fstar:.4f}, {Fp[5] - Fstar:.4f}, {Fp[10] - Fstar:.4f}")
with open(f"{OUT}/gd_estimators.dat", "w") as fh:
    fh.write("k sb sg tb tg\n")
    for k in range(81):
        fh.write(f"{k} {paths_e['spsa'][k, 0]:.6f} {paths_e['spsa'][k, 1]:.6f} {paths_e['two'][k, 0]:.6f} "
                 f"{paths_e['two'][k, 1]:.6f}\n")

"""Numerical data for the figures and the numbers quoted in Lecture 11 (continuous warm starts).

The worked example is a portfolio of n = 6 assets, as in Egger, Marecek and Woerner (2021): the prices follow
geometric Brownian motions for N = 250 days, with drifts uniform in [-5%, 5%] and volatilities uniform in
[-20%, 20%]; the daily returns give the mean mu and the covariance Sigma, both annualised (times 250). Choosing
B = 3 of the assets, with the risk aversion q = 2 and the penalty lambda = 3 for the budget, is the QUBO
    f(x) = x^T A x + b^T x + lambda B^2,  A = q Sigma + lambda 1 1^T,  b = -mu - 2 lambda B 1,
and the relaxation is f over [0, 1]^6, a convex quadratic program, solved exactly by enumerating the active sets.
The worked example is the first instance of the generator (seed 0). The circuits are simulated exactly: the warm
state is the product of R_Y(theta_i)|0>, theta_i = 2 arcsin sqrt(c_i), with c clipped to [eps, 1 - eps]; a layer
applies exp(-i gamma f) and the mixer exp(-i beta H_M), H_M = sum_i (-sin theta_i X_i - cos theta_i Z_i); eps = 0.5
gives standard QAOA, with |+>^6 and H_M = -sum_i X_i.

Writes, into figures/ch11/:
  ws_qubits.dat     i, c_i, theta_i, the regularised theta_i for eps = 0.25, the optimal bit d_i, and the points of
                    the warm states and of the regularised ones in the x-z cross-sections of the six Bloch spheres
  bloch.dat         the projected curves of the Bloch sphere of qubit 5: the equator, the meridian in the x-z plane,
                    the latitude of the warm state, and the orbit of the mixer, each split into front and back
  bloch_pts.dat     the projected points and axes of that sphere
  ws_dist.dat       for each of the 64 bit strings, f(x) - f(c*), its probability in the warm state and in |+>^6, and
                    whether it meets the budget
  ws_land0.png, ws_land25.png, ws_land50.png  the depth-one landscapes log10(F - F*) over (beta, gamma) for eps = 0,
                    0.25 and 0.5, as images for pgfplots
  ws_eps.dat        eps, the probability of the optimal portfolio and F - F* in the warm state and at the best
                    depth-one angles
  ws_depth.dat      p and, for standard QAOA, WS-QAOA with eps = 0 and with eps = 0.25, the quartiles of F - F* and
                    of the probability of the optimal portfolio over ten random starts of COBYLA
  ws_instances.dat  for 250 random instances, the overlap d*^T c* / B, F - F* of the best depth-one standard and
                    warm-started states, their ratio, and whether the optimum is reachable at eps = 0
  ws_anneal.dat     T and F - F* after the Trotterised annealing with N = 10 and N = 1000 steps, for eps = 0, 0.25, 0.5
and prints the numbers quoted in the text.
"""
import os, itertools
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "figures/ch11"
os.makedirs(OUT, exist_ok=True)
n, Bud, q, lam, Nd = 6, 3, 2.0, 3.0, 250
X = np.array([[(s >> (n - 1 - i)) & 1 for i in range(n)] for s in range(2**n)], float)

def instance(seed):
    g = np.random.default_rng(seed)
    mu, sg = g.uniform(-0.05, 0.05, n), g.uniform(-0.2, 0.2, n)
    W = np.cumsum(g.normal(size=(Nd, n)), 0) / np.sqrt(Nd)
    k = np.arange(1, Nd + 1)[:, None]
    S = np.vstack([np.ones(n), np.exp((mu - sg**2 / 2) * k / Nd + sg * W)])
    r = S[1:] / S[:-1] - 1
    return Nd * r.mean(0), Nd * np.cov(r.T)
def qubo(m, C):
    return q * C + lam * np.ones((n, n)), -m - 2 * lam * Bud * np.ones(n), lam * Bud**2
def fval(A, b, c0, x): return np.einsum("...i,ij,...j->...", x, A, x) + x @ b + c0
def relax(A, b):
    """The minimiser of x^T A x + b^T x over [0, 1]^n, A positive definite, by enumerating the active sets."""
    for st in itertools.product((2, 0, 1), repeat=n):           # 0 at the lower bound, 1 at the upper, 2 free
        st = np.array(st); F = st == 2; x = np.where(st == 1, 1.0, 0.0)
        if F.any():
            x[F] = np.linalg.solve(A[np.ix_(F, F)], -(b[F] / 2 + A[np.ix_(F, ~F)] @ x[~F]))
            if np.any(x[F] < -1e-12) or np.any(x[F] > 1 + 1e-12): continue
        gr = 2 * A @ x + b
        if np.all(gr[st == 0] >= -1e-10) and np.all(gr[st == 1] <= 1e-10):
            return np.clip(x, 0, 1)
    raise RuntimeError("no KKT point")

class Problem:
    def __init__(self, seed):
        self.m, self.C = instance(seed)
        self.A, self.b, self.c0 = qubo(self.m, self.C)
        self.c = relax(self.A, self.b)
        self.fx = fval(self.A, self.b, self.c0, X)
        self.iopt = int(np.argmin(self.fx)); self.d = X[self.iopt]; self.F0 = self.fx.min()
    def thetas(self, eps):
        return 2 * np.arcsin(np.sqrt(np.clip(self.c, eps, 1 - eps)))
    def state(self, th, betas, gammas):
        psi = np.ones(1)
        for t in th: psi = np.kron(psi, [np.cos(t / 2), np.sin(t / 2)])
        psi = psi.astype(complex)
        for be, ga in zip(betas, gammas):
            psi = psi * np.exp(-1j * ga * self.fx)
            P = psi.reshape((2,) * n)
            for i, t in enumerate(th):        # exp(-i beta H_M,i) = cos beta + i sin beta (sin t X + cos t Z)
                U = np.cos(be) * np.eye(2) + 1j * np.sin(be) * np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])
                P = np.moveaxis(np.tensordot(U, P, axes=([1], [i])), 0, i)
            psi = P.reshape(-1)
        return psi
    def evaluate(self, th, x):
        p = len(x) // 2
        pr = np.abs(self.state(th, x[:p], x[p:]))**2
        return pr @ self.fx - self.F0, pr[self.iopt]
    def best_depth1(self, th, nb=61, ng=121):
        """The least F - F* at depth one over beta in [0, pi), gamma in [-pi/3, pi/3]: the five best points of a grid,
        each refined by the Nelder-Mead method."""
        Bg, Gg = np.linspace(0, np.pi, nb, endpoint=False), np.linspace(-np.pi / 3, np.pi / 3, ng)
        L = np.array([[self.evaluate(th, np.array([be, ga]))[0] for ga in Gg] for be in Bg])
        best = None
        for ij in np.argsort(L, axis=None)[:5]:
            i, j = np.unravel_index(ij, L.shape)
            r = minimize(lambda x: self.evaluate(th, x)[0], np.array([Bg[i], Gg[j]]), method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 2000})
            if best is None or r.fun < best.fun: best = r
        return best.x, self.evaluate(th, best.x)

# --------------------------------------------------------------------------- the worked example
P = Problem(0)
assert np.allclose(fval(P.A, P.b, P.c0, X), np.einsum("si,ij,sj->s", X, q * P.C, X) - X @ P.m + lam * (X.sum(1) - Bud)**2)
gap = P.F0 - fval(P.A, P.b, P.c0, P.c)
print(f"the worked example: mu = ({', '.join(f'{x:.3f}' for x in P.m)}), the volatilities sqrt(diag Sigma) = "
      f"({', '.join(f'{x:.3f}' for x in np.sqrt(np.diag(P.C)))}); the relaxed optimum c* = ({', '.join(f'{x:.3f}' for x in P.c)}), "
      f"with sum {P.c.sum():.3f}; the optimal portfolio d* = {P.d.astype(int)}; F* = {P.F0:.4f}, f(c*) = "
      f"{fval(P.A, P.b, P.c0, P.c):.4f}, the integrality gap {gap:.4f}; f ranges over [{P.fx.min():.3f}, {P.fx.max():.3f}]")
feas = X.sum(1) == Bud
print(f"  {feas.sum()} portfolios meet the budget; the second best has F - F* = {np.sort(P.fx[feas])[1] - P.F0:.4f}, "
      f"the worst feasible {P.fx[feas].max() - P.F0:.4f}; every other string has F - F* >= {P.fx[~feas].min() - P.F0:.3f}")

# the six qubits: x-z cross-sections, the point (sin theta, cos theta) relative to the centre of each circle
th0, th25 = P.thetas(0.0), P.thetas(0.25)
with open(f"{OUT}/ws_qubits.dat", "w") as fh:
    fh.write("i c th th25 d cx px pz qx qz\n")
    for i in range(n):
        cx = 2.7 * i
        fh.write(f"{i + 1} {P.c[i]:.4f} {th0[i]:.4f} {th25[i]:.4f} {int(P.d[i])} {cx:.2f} {cx + np.sin(th0[i]):.4f} "
                 f"{np.cos(th0[i]):.4f} {cx + np.sin(th25[i]):.4f} {np.cos(th25[i]):.4f}\n")
print(f"  theta_i = ({', '.join(f'{x:.3f}' for x in th0)}); with eps = 0.25, ({', '.join(f'{x:.3f}' for x in th25)}); "
      f"frozen at eps = 0: qubits {[i + 1 for i in range(n) if P.c[i] in (0.0, 1.0)]}")

# the Bloch sphere of qubit 5, orthographic, azimuth 30 degrees and elevation 18 degrees
az, el = np.radians(30), np.radians(18)
dv = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
eu = np.array([-np.sin(az), np.cos(az), 0.0])
ev = np.cross(dv, eu)
def proj(p): p = np.atleast_2d(p); return p @ eu, p @ ev, p @ dv
def split(pts):
    u, v, w = proj(pts)
    fu, fv = np.where(w >= 0, u, np.nan), np.where(w >= 0, v, np.nan)
    bu, bv = np.where(w < 0, u, np.nan), np.where(w < 0, v, np.nan)
    return fu, fv, bu, bv
t = np.linspace(0, 2 * np.pi, 361)
iq = 4
thq = th0[iq]; nvec = np.array([np.sin(thq), 0, np.cos(thq)])
phi = 1.1                                                       # a rotation about z by the cost layer
p1 = np.array([np.sin(thq) * np.cos(phi), np.sin(thq) * np.sin(phi), np.cos(thq)])
def rot(v, axis, a):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * (axis @ v) * (1 - np.cos(a))
curves = {"eq": np.stack([np.cos(t), np.sin(t), 0 * t], 1),
          "mer": np.stack([np.sin(t), 0 * t, np.cos(t)], 1),
          "lat": np.stack([np.sin(thq) * np.cos(t), np.sin(thq) * np.sin(t), np.cos(thq) + 0 * t], 1),
          "orb": np.array([rot(p1, nvec, a) for a in t])}
with open(f"{OUT}/bloch.dat", "w") as fh:
    cols = []
    for k_ in curves: cols += [f"{k_}fu", f"{k_}fv", f"{k_}bu", f"{k_}bv"]
    fh.write(" ".join(cols) + "\n")
    data = [split(curves[k_]) for k_ in curves]
    for j in range(len(t)):
        fh.write(" ".join("nan" if np.isnan(a[j]) else f"{a[j]:.5f}" for d4 in data for a in d4) + "\n")
pts = {"zero": [0, 0, 1], "one": [0, 0, -1], "warm": nvec, "anti": -nvec, "p1": p1, "x": [1, 0, 0], "y": [0, 1, 0],
       "hgt": [0, 0, np.cos(thq)], "e25": [0, 0, 0.5], "e75": [0, 0, -0.5], "orbtop": rot(p1, nvec, 2.2)}
with open(f"{OUT}/bloch_pts.dat", "w") as fh:
    fh.write("name u v w\n")
    for k_, p_ in pts.items():
        u, v, w = proj(np.array(p_, float))
        fh.write(f"{k_} {u[0]:.5f} {v[0]:.5f} {w[0]:.5f}\n")
print(f"  the Bloch sphere of qubit 5: c = {P.c[iq]:.3f}, theta = {thq:.3f}, the height cos theta = 1 - 2c = {np.cos(thq):.3f}; "
      f"the cost layer turns it by phi = {phi} about z, and the mixer then turns it on a circle about the axis through "
      f"the warm state, of angular radius {np.degrees(np.arccos(p1 @ nvec)):.1f} degrees")

# the warm distribution at depth zero
pw = np.prod(np.where(X == 1, P.c, 1 - P.c), axis=1)
pu = np.full(2**n, 2.0**-n)
fc = fval(P.A, P.b, P.c0, P.c)
var = np.sum(np.diag(P.A) * P.c * (1 - P.c))
assert abs(pw @ P.fx - (fc + var)) < 1e-10
with open(f"{OUT}/ws_dist.dat", "w") as fh:
    fh.write("s fx pw pu feas\n")
    for s in range(2**n):
        fh.write(f"{s} {P.fx[s] - fc:.6f} {pw[s]:.6f} {pu[s]:.6f} {int(feas[s])}\n")
print(f"the warm state at depth zero: E f = f(c*) + sum_i A_ii c_i (1 - c_i) = {fc:.4f} + {var:.4f} = {pw @ P.fx:.4f}, "
      f"F - F* = {pw @ P.fx - P.F0:.4f}; of the variance term, lambda sum c(1-c) = {lam * np.sum(P.c * (1 - P.c)):.4f} and "
      f"q sum Sigma_ii c(1-c) = {q * np.sum(np.diag(P.C) * P.c * (1 - P.c)):.4f}; the optimum has probability {pw[P.iopt]:.4f}, "
      f"the budget is met with probability {pw[feas].sum():.4f}, {int(np.sum(pw > 0))} strings have positive probability; "
      f"|+>^6: E f - F* = {pu @ P.fx - P.F0:.4f}, the optimum {pu[P.iopt]:.4f}, the budget {pu[feas].sum():.4f}")

# --------------------------------------------------------------------------- the landscapes at depth one
cmap = LinearSegmentedColormap.from_list("ylgnbu", [np.array(c_) / 255 for c_ in
       [(12, 44, 132), (34, 94, 168), (29, 145, 192), (65, 182, 196), (127, 205, 187), (199, 233, 180), (255, 255, 204)]])
lo_, hi_ = -0.30103, 1.30103                                    # the colour scale of log10(F - F*), from 0.5 to 20
Bf, Gf = np.linspace(0, np.pi, 241), np.linspace(-np.pi / 3, np.pi / 3, 241)
land = {}
for eps, tag in ((0.0, "0"), (0.25, "25"), (0.5, "50")):
    th = P.thetas(eps)
    L = np.array([[P.evaluate(th, np.array([be, ga]))[0] for be in Bf] for ga in Gf])
    plt.imsave(f"{OUT}/ws_land{tag}.png", np.clip(np.log10(L), lo_, hi_), cmap=cmap, vmin=lo_, vmax=hi_, origin="lower")
    xb, (eb, pb) = P.best_depth1(th)
    land[tag] = (L, xb, eb, pb, P.evaluate(th, np.zeros(2)))
    print(f"  landscape eps = {eps}: F - F* from {L.min():.4f} to {L.max():.3f} on the grid; at zero angles "
          f"{land[tag][4][0]:.4f}, the optimum with probability {land[tag][4][1]:.4f}; the best depth-one angles "
          f"beta = {xb[0]:.3f}, gamma = {xb[1]:.3f}, F - F* = {eb:.4f}, probability {pb:.4f}")
with open(f"{OUT}/ws_landmin.dat", "w") as fh:
    fh.write("tag beta gamma\n")
    for tag in ("0", "25", "50"):
        fh.write(f"{tag} {land[tag][1][0] % np.pi:.4f} {land[tag][1][1]:.4f}\n")

# the best depth-one state against eps
with open(f"{OUT}/ws_eps.dat", "w") as fh:
    fh.write("eps p0 e0 p1 e1\n")
    for eps in np.linspace(0, 0.5, 11):
        th = P.thetas(eps)
        e0, p0 = P.evaluate(th, np.zeros(2)); _, (e1, p1_) = P.best_depth1(th, nb=41, ng=81)
        fh.write(f"{eps:.3f} {p0:.5f} {e0:.5f} {p1_:.5f} {e1:.5f}\n")
        print(f"  eps = {eps:.2f}: the warm state F - F* = {e0:.4f}, optimum {p0:.4f}; depth one F - F* = {e1:.4f}, optimum {p1_:.4f}")

# --------------------------------------------------------------------------- depth 1 to 5, ten random starts of COBYLA
rng_d = np.random.default_rng(1101)
rows = []
for p in range(1, 6):
    row = [p]
    for eps in (0.5, 0.0, 0.25):
        th = P.thetas(eps); res = []
        for s in range(10):
            r = minimize(lambda x: P.evaluate(th, x)[0], rng_d.uniform(-2 * np.pi, 2 * np.pi, 2 * p), method="COBYLA",
                         options={"rhobeg": 0.5, "maxiter": 3000, "tol": 1e-6})
            res.append(P.evaluate(th, r.x))
        res = np.array(res)
        row += list(np.percentile(res[:, 0], [25, 50, 75])) + list(np.percentile(res[:, 1], [25, 50, 75]))
    rows.append(row)
    print(f"  depth {p}: median F - F* {row[2]:.4f} (standard), {row[8]:.4f} (eps = 0), {row[14]:.4f} (eps = 0.25); "
          f"median probability of the optimum {row[5]:.3f}, {row[11]:.3f}, {row[17]:.3f}")
with open(f"{OUT}/ws_depth.dat", "w") as fh:
    names = []
    for tag in ("s", "w", "r"): names += [f"{tag}e25", f"{tag}e50", f"{tag}e75", f"{tag}p25", f"{tag}p50", f"{tag}p75"]
    fh.write("p " + " ".join(names) + "\n")
    for row in rows: fh.write(" ".join(f"{x:.6f}" for x in row) + "\n")

# --------------------------------------------------------------------------- 250 random instances at depth one
inst = []
for seed in range(1, 251):
    Q = Problem(seed)
    _, (ec, pc) = Q.best_depth1(Q.thetas(0.5), nb=31, ng=61)
    _, (ew, pw_) = Q.best_depth1(Q.thetas(0.0), nb=31, ng=61)
    frozen = (Q.c == 0) | (Q.c == 1)
    reach = bool(np.all(Q.d[frozen] == Q.c[frozen]))
    inst.append((seed, Q.d @ Q.c / Bud, ec, ew, ew / ec, int(reach), pc, pw_))
inst = np.array(inst)
with open(f"{OUT}/ws_instances.dat", "w") as fh:
    fh.write("seed overlap ecold ewarm ratio reach pcold pwarm\n")
    for r_ in inst: fh.write(f"{int(r_[0])} {r_[1]:.4f} {r_[2]:.5f} {r_[3]:.5f} {r_[4]:.5f} {int(r_[5])} {r_[6]:.4f} {r_[7]:.4f}\n")
rc = inst[:, 5] == 1
print(f"250 instances at depth one: the warm start lowers F - F* on {np.mean(inst[:, 4] < 1) * 100:.0f}% of them; median "
      f"F - F* {np.median(inst[:, 2]):.3f} (standard) and {np.median(inst[:, 3]):.3f} (warm); median ratio "
      f"{np.median(inst[:, 4]):.3f}; median probability of the optimum {np.median(inst[:, 6]):.3f} and {np.median(inst[:, 7]):.3f}; "
      f"the optimum is unreachable at eps = 0 on {int(np.sum(~rc))}; the ratio is below 0.1 on "
      f"{int(np.sum(inst[:, 4] < 0.1))}, of which {int(np.sum((inst[:, 4] < 0.1) & (inst[:, 1] > 0.9)))} have the overlap above 0.9")

# --------------------------------------------------------------------------- Trotterised annealing
def anneal(th, T, N):
    k = np.arange(1, N + 1); dt = T / N
    return P.evaluate(th, np.concatenate([dt * (1 - k / N), dt * k / N]))
Ts = np.exp(np.linspace(np.log(0.1), np.log(50), 41))
with open(f"{OUT}/ws_anneal.dat", "w") as fh:
    fh.write("T " + " ".join(f"e{tag}n{N}" for tag in ("0", "25", "50") for N in (10, 1000)) + "\n")
    tab = {}
    for eps, tag in ((0.0, "0"), (0.25, "25"), (0.5, "50")):
        th = P.thetas(eps)
        for N in (10, 1000): tab[(tag, N)] = [anneal(th, T, N)[0] for T in Ts]
    for j, T in enumerate(Ts):
        fh.write(f"{T:.4f} " + " ".join(f"{tab[(tag, N)][j]:.6f}" for tag in ("0", "25", "50") for N in (10, 1000)) + "\n")
for tag, nm in (("0", "eps = 0"), ("25", "eps = 0.25"), ("50", "standard")):
    e_ = np.array(tab[(tag, 1000)])
    k1 = np.argmax(e_ <= 0.1)
    print(f"  annealing, {nm}, N = 1000: F - F* = {np.interp(5, Ts, e_):.4f} at T = 5, {e_[-1]:.4f} at T = 50; below 0.1 from "
          f"T = {Ts[k1]:.2f}; with N = 10 at T = 5: {np.interp(5, Ts, tab[(tag, 10)]):.4f}, at T = 50: {tab[(tag, 10)][-1]:.4f}")
th = P.thetas(0.0)
print(f"  annealing, eps = 0, N = 1000, T = 50: the optimum has probability {anneal(th, 50, 1000)[1]:.3f}")

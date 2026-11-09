"""Numerical data for the figures and the numbers quoted in Lecture 13 (Markowitz portfolios).

The worked example is the portfolio of Lecture 11: the first instance (seed 0) of the generator of Egger, Marecek and
Woerner (2021), n = 6 assets whose prices follow geometric Brownian motions for 250 days, with drifts uniform in
[-5%, 5%] and volatilities uniform in [-20%, 20%]; mu and Sigma are the mean and the covariance of the daily returns,
annualised. Choosing B = 3 assets with the risk aversion q = 2, the objective of a selection x in {0, 1}^6 is
    g(x) = q x^T Sigma x - mu^T x,
and the penalised QUBO is f(x) = g(x) + lambda (1^T x - B)^2 with lambda = 3. The relaxation over the budget polytope
P = {x in [0, 1]^6 : 1^T x = B} and the frontier of P are computed exactly, by enumerating the active sets.

The solvers are simulated exactly: standard QAOA and the continuous warm start of Lecture 11 on the 64 strings, with the
phase separator f; QAOA trained on the CVaR of f with alpha = 0.1; and the alternating operator ansatz on the 20
feasible strings, from the Dicke state, with the ring XY mixer, the complete XY mixer and the Grover mixer, and the
phase separator g. At depth one, the five best points of a grid over a period of beta and gamma in [-10 pi, 10 pi] are
refined by the Nelder-Mead method; at depth p > 1, BFGS starts from each of the five best angles of depth p - 1 with a
layer of zero angles appended, from a perturbation of it, and from its INTERP extension (Zhou et al. 2020), and from
ten random points with gamma in [-pi/3, pi/3] and ten with gamma in [-10 pi, 10 pi]; the best of the 35 is kept. On the
worked example, a hundred random points of each kind are used instead, and the best of 215 is kept.

Writes, into figures/ch13/:
  frontier.dat       the boundary of the risk-return region of P: risk sqrt(x^T Sigma x), return mu^T x, efficient
  frontier_pts.dat   the 20 selections with B ones: risk, return, g - F*, whether non-dominated, whether optimal
  penalty.dat        the 64 strings: the number of ones k, a jittered abscissa, f - F* with lambda = 3, g - F*, feasible
  threshold.dat      lambda and, for k = 0, ..., 6, the least value of g + lambda (k - B)^2 over the strings with k ones
  johnson_nodes.dat  the 20 feasible strings in a layered drawing of the Johnson graph J(6, 3): x, y, g - F*, layer
  johnson_edges.dat  the edges of J(6, 3) that the ring XY mixer lacks, as segments separated by nan rows
  johnson_ring.dat   the edges of the ring XY mixer
  mp_hist.dat        the histogram of the eigenvalues of a sample covariance, N = 600, T = 2500, true covariance 1
  mp_curve.dat       the Marchenko-Pastur density with q = N/T
  gmv.dat            N, q = N/T and the in-sample and out-of-sample variances of the sample minimum-variance portfolio
                     relative to the true least variance, T = 250, the mean and the quartiles of 1000 samples
  frontiers.dat      the true efficient frontier of a one-factor market of N = 10 assets
  frontiers_in.dat   twenty in-sample frontiers estimated from T = 1250 days, separated by nan rows
  frontiers_out.dat  the true risk and return of the same portfolios
  resample.dat       the 20 portfolios of the worked example in the order of their true objective, with the frequency
                     with which each is optimal for the estimates from 4000 fresh samples of 250 days
  solvers_depth.dat  p = 0, ..., 5 and, for each solver, F - F*, the probability of the optimal portfolio and of the
                     budget on the worked example
  solvers_inst.dat   p = 1, ..., 5 and, for each solver, the quartiles of the probability of the optimal portfolio over
                     100 random instances
and prints the numbers quoted in the text. It takes thirty to forty-five minutes, with eight processes.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import itertools
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb

OUT = "figures/ch13"
n, Bud, q, lam, Nd = 6, 3, 2.0, 3.0, 250
X = np.array([[(s >> (n - 1 - i)) & 1 for i in range(n)] for s in range(2**n)], float)
wt = X.sum(1)
FE = np.where(wt == Bud)[0]
Fx = X[FE]
nf = len(FE)
lab = lambda x: "".join(str(int(v)) for v in x)

def instance(seed):
    g = np.random.default_rng(seed)
    mu, sg = g.uniform(-0.05, 0.05, n), g.uniform(-0.2, 0.2, n)
    W = np.cumsum(g.normal(size=(Nd, n)), 0) / np.sqrt(Nd)
    k = np.arange(1, Nd + 1)[:, None]
    S = np.vstack([np.ones(n), np.exp((mu - sg**2 / 2) * k / Nd + sg * W)])
    r = S[1:] / S[:-1] - 1
    return Nd * r.mean(0), Nd * np.cov(r.T)
def gvals(m, C, Y=X): return np.einsum("si,ij,sj->s", Y, q * C, Y) - Y @ m

def boxqp(Q, c, E, e):
    """The minimiser of x^T Q x + c^T x subject to E x = e and 0 <= x <= 1, Q positive definite, by enumerating the
    active sets and checking the conditions of Karush, Kuhn and Tucker."""
    nn, me = len(c), len(e)
    for st in itertools.product((2, 0, 1), repeat=nn):           # 2 free, 0 at the lower bound, 1 at the upper
        st = np.array(st); F = st == 2; x = np.where(st == 1, 1.0, 0.0)
        nF = F.sum()
        K = np.zeros((nF + me, nF + me))
        K[:nF, :nF] = 2 * Q[np.ix_(F, F)]; K[:nF, nF:] = E[:, F].T; K[nF:, :nF] = E[:, F]
        rhs = np.concatenate([-c[F] - 2 * Q[np.ix_(F, ~F)] @ x[~F], e - E[:, ~F] @ x[~F]])
        if nF + me and np.linalg.matrix_rank(K) < nF + me: continue
        sol = np.linalg.solve(K, rhs) if nF + me else np.zeros(0)
        x[F] = sol[:nF]; nu = sol[nF:]
        if np.any(x[F] < -1e-10) or np.any(x[F] > 1 + 1e-10): continue
        gr = 2 * Q @ x + c + E.T @ nu
        if np.all(gr[st == 0] >= -1e-9) and np.all(gr[st == 1] <= 1e-9):
            return np.clip(x, 0, 1)
    return None

def relax_pen(A, b):
    """The minimiser of x^T A x + b^T x over [0, 1]^n (Lecture 11)."""
    return boxqp(A, b, np.zeros((0, n)), np.zeros(0))

# --------------------------------------------------------------------------- the circuits
Ac = np.array([[1.0 if np.abs(a - b).sum() == 2 else 0.0 for b in Fx] for a in Fx])
def ring_edge(a, b):
    d = np.where(a != b)[0]
    return len(d) == 2 and (d[1] - d[0] == 1 or (d[0] == 0 and d[1] == n - 1))
Ar = np.array([[1.0 if ring_edge(a, b) else 0.0 for b in Fx] for a in Fx])
ec, Vc = np.linalg.eigh(Ac)
er, Vr = np.linalg.eigh(Ar)
Dk = np.ones(nf) / np.sqrt(nf)
def product_mixer(th):
    """The eigenvectors V and eigenvalues s of sum_i (sin th_i X_i + cos th_i Z_i); the layer of the mixer of Lecture 11 is
    exp(i beta sum_i n_i.sigma) = V diag(exp(i beta s)) V^T."""
    V, s = np.ones((1, 1)), np.zeros(1)
    for t in th:
        w_, v_ = np.linalg.eigh(np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]]))
        V, s = np.kron(V, v_), np.add.outer(s, w_).ravel()
    return V, s

class Solver:
    """A circuit: initial state psi0, phase separator exp(-i gamma cost), mixer V diag(exp(i beta s)) V^T, and the
    objective, the expectation of cost ('E') or its CVaR at the level alpha (a number)."""
    def __init__(self, name, cost, psi0, V, s, per, obj, iopt, feas, F0):
        self.name, self.cost, self.psi0, self.V, self.s, self.per = name, cost, psi0.astype(complex), V, s, per
        self.obj, self.iopt, self.feas, self.F0 = obj, iopt, feas, F0
        self.order = np.argsort(cost)
    def probs(self, x):
        p = len(x) // 2; psi = self.psi0
        for be, ga in zip(x[:p], x[p:]):
            psi = self.V @ (np.exp(1j * be * self.s) * (self.V.T @ (np.exp(-1j * ga * self.cost) * psi)))
        return np.abs(psi)**2
    def value(self, P):
        """The objective of the distributions in the columns of P (or of one distribution)."""
        if self.obj == "E": return self.cost @ P
        a = self.obj; Ps = P[self.order]
        before = np.cumsum(Ps, 0) - Ps
        take = np.minimum(Ps, np.maximum(a - before, 0))
        return self.cost[self.order] @ take / a
    def f(self, x): return self.value(self.probs(x))

def solvers(m, C, alpha=0.1):
    g = gvals(m, C); fl = g + lam * (wt - Bud)**2
    A = q * C + lam * np.ones((n, n)); b = -m - 2 * lam * Bud * np.ones(n)
    c = relax_pen(A, b)
    thw = 2 * np.arcsin(np.sqrt(c))
    psiw = np.ones(1)
    for t in thw: psiw = np.kron(psiw, [np.cos(t / 2), np.sin(t / 2)])
    V2, s2 = product_mixer(np.full(n, np.pi / 2)); Vw, sw = product_mixer(thw)
    io_full, io_feas = int(np.argmin(fl)), int(np.argmin(g[FE]))
    assert FE[io_feas] == io_full
    fe_full, fe_feas = wt == Bud, np.ones(nf, bool); F0 = fl.min()
    gf = g[FE]
    out = [Solver("standard", fl, np.ones(64) / 8, V2, s2, np.pi, "E", io_full, fe_full, F0),
           Solver("warm", fl, psiw, Vw, sw, np.pi, "E", io_full, fe_full, F0),
           Solver("cvar", fl, np.ones(64) / 8, V2, s2, np.pi, alpha, io_full, fe_full, F0),
           Solver("ring", gf, Dk, Vr, -er, 2 * np.pi, "E", io_feas, fe_feas, F0),
           Solver("complete", gf, Dk, Vc, -ec, np.pi, "E", io_feas, fe_feas, F0),
           Solver("grover", gf, Dk, np.column_stack([Dk, np.linalg.svd(np.eye(nf) - np.outer(Dk, Dk))[0][:, :nf - 1]]),
                  -np.r_[1.0, np.zeros(nf - 1)], 2 * np.pi, "E", io_feas, fe_feas, F0)]
    return out, c, g, fl

GAM = 10 * np.pi
def depth1(S, nb=72, ng=2001):
    Bg = np.linspace(0, S.per, nb, endpoint=False); Gg = np.linspace(-GAM, GAM, ng)
    Y = S.V.T @ (np.exp(-1j * np.outer(S.cost, Gg)) * S.psi0[:, None])
    L = np.array([S.value(np.abs(S.V @ (np.exp(1j * be * S.s)[:, None] * Y))**2) for be in Bg])
    res = []
    for ij in np.argsort(L, axis=None)[:5]:
        i, j = np.unravel_index(ij, L.shape)
        r = minimize(S.f, np.array([Bg[i], Gg[j]]), method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 4000})
        res.append((r.fun, r.x))
    return sorted(res, key=lambda t: t[0])
def interp(x):
    p = len(x) // 2
    def up(v):
        return np.array([(i / p) * (v[i - 1] if i >= 1 else 0.0) + ((p - i) / p) * (v[i] if i < p else 0.0)
                         for i in range(p + 1)])
    return np.concatenate([up(x[:p]), up(x[p:])])
def optimise(S, pmax, rng, nrand=10):
    prev = depth1(S); out = [prev[0]]
    for p in range(2, pmax + 1):
        starts = []
        for _, b0 in prev[:5]:
            pad = np.concatenate([b0[:p - 1], [0.0], b0[p - 1:], [0.0]])
            starts += [pad, pad + rng.normal(0, 0.1, 2 * p), interp(b0)]
        starts += [np.concatenate([rng.uniform(0, S.per, p), rng.uniform(-G, G, p)])
                   for G in (np.pi / 3, GAM) for _ in range(nrand)]
        res = []
        for x0 in starts:
            r = minimize(S.f, x0, method="BFGS", options={"gtol": 1e-10, "maxiter": 4000})
            res.append((min(r.fun, S.f(x0)), r.x if r.fun <= S.f(x0) else x0))
        prev = sorted(res, key=lambda t: t[0]); out.append(prev[0])
    return out

NAMES = ["standard", "warm", "cvar", "ring", "complete", "grover"]
PMAX = 5
def run_instance(seed):
    m, C = instance(seed)
    Ss, c, g, fl = solvers(m, C)
    rng = np.random.default_rng(1300 + seed)
    row = {}
    for S in Ss:
        res = optimise(S, PMAX, rng)
        row[S.name] = [(S.probs(x) @ S.cost - S.F0, S.probs(x)[S.iopt], S.probs(x)[S.feas].sum()) for _, x in res]
        row[S.name + "0"] = (S.psi0.real**2 @ S.cost - S.F0, abs(S.psi0[S.iopt])**2)
    d = X[int(np.argmin(fl))]
    cB = boxqp(q * C, -m, np.ones((1, n)), np.array([float(Bud)]))
    rnd = np.zeros(n); rnd[np.argsort(-cB)[:Bud]] = 1
    row["round"] = bool(np.all(rnd == d))
    row["overlap"] = d @ c / Bud
    return seed, row

def segments(edges, P):
    rows = []
    for i, j in edges: rows += [P[i], P[j], (np.nan, np.nan)]
    return rows

def main():
    os.makedirs(OUT, exist_ok=True)
    # ----------------------------------------------------------------------- the worked example
    m, C = instance(0)
    g = gvals(m, C); fl = g + lam * (wt - Bud)**2
    gf = g[FE]; F0 = gf.min(); iopt = int(np.argmin(gf)); dstar = Fx[iopt]
    srt = np.sort(gf)
    print(f"the worked example: d* = {lab(dstar)}, F* = {F0:.4f}; the 20 portfolios span {gf.max() - F0:.4f} above F*, "
          f"the second best {srt[1] - F0:.4f} above, the five best within {srt[4] - F0:.4f}")
    risk, ret = np.sqrt(np.einsum("si,ij,sj->s", Fx, C, Fx)), Fx @ m
    nd = [all(not ((ret[j] >= ret[i]) and (risk[j] <= risk[i]) and ((ret[j] > ret[i]) or (risk[j] < risk[i])))
              for j in range(nf)) for i in range(nf)]
    one = np.ones((1, n))
    cB = boxqp(q * C, -m, one, np.array([float(Bud)]))
    print(f"  the relaxation over P: c_B = ({', '.join(f'{v:.3f}' for v in cB)}), g(c_B) = {cB @ (q * C) @ cB - m @ cB:.4f}, "
          f"risk {np.sqrt(cB @ C @ cB):.4f}, return {m @ cB:.4f}; d*: risk {risk[iopt]:.4f}, return {ret[iopt]:.4f}")
    rnd = np.zeros(n); rnd[np.argsort(-cB)[:Bud]] = 1
    print(f"  rounding c_B to its {Bud} largest coordinates gives {lab(rnd)}, {gvals(m, C, rnd[None, :])[0] - F0:.4f} above F*")
    print(f"  non-dominated selections: {[(lab(Fx[i]), round(risk[i], 3), round(ret[i], 3)) for i in range(nf) if nd[i]]}; "
          f"d* has the largest return of all {nf}: {ret[iopt] >= ret.max() - 1e-12}")
    rlo, rhi = np.sort(m)[:Bud].sum(), np.sort(m)[-Bud:].sum()
    xmv = boxqp(C, np.zeros(n), one, np.array([float(Bud)]))
    rows = []
    for r in np.linspace(rlo, rhi, 241):
        x = boxqp(C, np.zeros(n), np.vstack([one, m[None, :]]), np.array([float(Bud), r]))
        rows.append((np.sqrt(x @ C @ x), r, int(r >= m @ xmv - 1e-12)))
    with open(f"{OUT}/frontier.dat", "w") as fh:
        fh.write("risk ret eff\n")
        for r in rows: fh.write(f"{r[0]:.5f} {r[1]:.5f} {r[2]}\n")
    print(f"  the frontier of P: returns from {rlo:.4f} to {rhi:.4f}; the least risk {np.sqrt(xmv @ C @ xmv):.4f} at the "
          f"return {m @ xmv:.4f}")
    with open(f"{OUT}/frontier_pts.dat", "w") as fh:
        fh.write("risk ret dg nd opt label\n")
        for i in range(nf):
            fh.write(f"{risk[i]:.5f} {ret[i]:.5f} {gf[i] - F0:.5f} {int(nd[i])} {int(i == iopt)} {lab(Fx[i])}\n")

    # the penalty
    with open(f"{OUT}/penalty.dat", "w") as fh:
        fh.write("k xj df dg feas\n")
        for k in range(n + 1):
            idx = np.where(wt == k)[0]; idx = idx[np.argsort(g[idx])]
            for r, s in enumerate(idx):
                xj = k + (0.0 if len(idx) == 1 else -0.32 + 0.64 * r / (len(idx) - 1))
                fh.write(f"{k} {xj:.4f} {fl[s] - F0:.5f} {g[s] - F0:.5f} {int(k == Bud)}\n")
    mk = np.array([g[wt == k].min() for k in range(n + 1)])
    thr = [((F0 - mk[k]) / (k - Bud)**2, k) for k in range(n + 1) if k != Bud]
    lstar, kstar = max(thr)
    Delta = max(abs(g[s ^ (1 << (n - 1 - i))] - g[s]) for s in range(2**n) for i in range(n))
    bound = max(abs(m[i]) + q * C[i, i] + 2 * q * (np.abs(C[i]).sum() - abs(C[i, i])) for i in range(n))
    print(f"  the least g over k ones: {', '.join(f'{v:.4f}' for v in mk)}; the exact threshold lambda* = {lstar:.4f}, set "
          f"by k = {kstar}; the thresholds of the other k: {', '.join(f'k={k}: {t:.4f}' for t, k in thr)}")
    print(f"  Delta = {Delta:.4f}, its bound max_i |mu_i| + q Sigma_ii + 2q sum_j |Sigma_ij| = {bound:.4f}; lambda = {lam} is "
          f"{lam / lstar:.0f} times lambda*, and Delta is {Delta / lstar:.0f} times lambda*; with it f - F* ranges up to {fl.max() - F0:.2f}, every infeasible string at least "
          f"{fl[wt != Bud].min() - F0:.3f} above F*")
    with open(f"{OUT}/threshold.dat", "w") as fh:
        fh.write("lam " + " ".join(f"k{k}" for k in range(n + 1)) + "\n")
        for L in np.linspace(0, 0.08, 161):
            fh.write(f"{L:.5f} " + " ".join(f"{mk[k] + L * (k - Bud)**2:.5f}" for k in range(n + 1)) + "\n")

    # the Johnson graph J(6, 3), layered by the distance from d*
    layer = ((Fx != dstar).sum(1) // 2).astype(int)
    P = np.zeros((nf, 2))
    for L in range(4):
        idx = np.where(layer == L)[0]; idx = idx[np.argsort(gf[idx])]
        for r, i in enumerate(idx):
            xx = 0.0 if len(idx) == 1 else (r - (len(idx) - 1) / 2) * 1.25
            P[i] = (xx, (3 - L) * 2.6 + 0.08 * xx**2 + (0.6 if L == 0 else 0.0))
    E_all = [(i, j) for i in range(nf) for j in range(i + 1, nf) if Ac[i, j]]
    E_ring = [(i, j) for i, j in E_all if Ar[i, j]]
    with open(f"{OUT}/johnson_nodes.dat", "w") as fh:
        fh.write("x y dg layer ringdeg opt label\n")
        for i in range(nf):
            fh.write(f"{P[i, 0]:.4f} {P[i, 1]:.4f} {gf[i] - F0:.5f} {layer[i]} {int(Ar[i].sum())} {int(i == iopt)} {lab(Fx[i])}\n")
    for fn, E in (("johnson_edges.dat", [e for e in E_all if e not in E_ring]), ("johnson_ring.dat", E_ring)):
        with open(f"{OUT}/{fn}", "w") as fh:
            fh.write("x y\n")
            for x, y in segments(E, P): fh.write(f"{x:.4f} {y:.4f}\n")
    print(f"  J(6, 3): {len(E_all)} edges, {len(E_ring)} of them in the ring XY mixer; layers of sizes "
          f"{[int((layer == L).sum()) for L in range(4)]}; the degrees in the ring mixer {sorted(set(Ar.sum(1).astype(int)))}")
    vals, cnts = np.unique(np.round(ec, 8), return_counts=True)
    print(f"  the spectrum of the complete XY mixer on the 20 portfolios: {', '.join(f'{v:g} (x{c})' for v, c in zip(vals[::-1], cnts[::-1]))}")
    for j in range(4):
        print(f"    j = {j}: (B-j)(n-B-j)-j = {(Bud - j) * (n - Bud - j) - j}, multiplicity C(n,j)-C(n,j-1) = "
              f"{comb(n, j, exact=True) - (comb(n, j - 1, exact=True) if j else 0)}")
    top = Vr[:, -1] * np.sign(Vr[0, -1])
    vals, cnts = np.unique(np.round(er, 8), return_counts=True)
    print(f"  the ring XY mixer: the spectrum {', '.join(f'{v:g} (x{c})' for v, c in zip(vals[::-1], cnts[::-1]))}; its top "
          f"eigenvector has entries from {top.min():.4f} to {top.max():.4f} and the overlap {abs(top @ Dk):.4f} with the Dicke state")
    path, kk, pr = [], Bud, 1.0
    for mm in range(n, 0, -1):
        bit = int(dstar[mm - 1]); p_ = kk / mm if bit else (mm - kk) / mm
        path.append(f"x{mm}={bit}: {p_:.4f}"); pr *= p_; kk -= bit
    print(f"  Dicke preparation along d*: {'; '.join(path)}; the product {pr:.6f} = 1/{1 / pr:.0f}")

    # ----------------------------------------------------------------------- estimation error
    rng = np.random.default_rng(1301)
    Nm, Tm = 600, 2500; qm = Nm / Tm
    ev = np.linalg.eigvalsh(np.cov(rng.normal(size=(Tm, Nm)).T))
    lm, lp = (1 - np.sqrt(qm))**2, (1 + np.sqrt(qm))**2
    h, edges = np.histogram(ev, bins=42, range=(0.1, 2.4), density=True)
    with open(f"{OUT}/mp_hist.dat", "w") as fh:
        fh.write("x dens\n")
        for x, d in zip((edges[1:] + edges[:-1]) / 2, h): fh.write(f"{x:.4f} {d:.5f}\n")
    with open(f"{OUT}/mp_curve.dat", "w") as fh:
        fh.write("x dens\n")
        for x in np.linspace(lm, lp, 301):
            fh.write(f"{x:.5f} {np.sqrt(max((lp - x) * (x - lm), 0)) / (2 * np.pi * qm * x):.5f}\n")
    print(f"Marchenko-Pastur, N = {Nm}, T = {Tm}, q = {qm}: the edges {lm:.4f} and {lp:.4f}; the sample eigenvalues from "
          f"{ev.min():.4f} to {ev.max():.4f}, all equal to 1 in truth; the condition number {ev.max() / ev.min():.2f}; along the "
          f"eigenvector of the least, the variance is underestimated by the factor {1 / ev.min():.2f}")

    rows = []
    for N in (5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225):
        T = 250; ins, outs = [], []
        for _ in range(1000):
            Si = np.linalg.inv(np.cov(rng.normal(size=(T, N)).T)); w_ = Si.sum(1) / Si.sum()
            ins.append(N / Si.sum()); outs.append(N * (w_ @ w_))
        rows.append((N, N / T, np.mean(ins), *np.percentile(ins, [25, 75]), (T - N) / (T - 1), np.mean(outs),
                     *np.percentile(outs, [25, 75])))
    with open(f"{OUT}/gmv.dat", "w") as fh:
        fh.write("N q in in25 in75 inexact out out25 out75\n")
        for r in rows: fh.write(" ".join(f"{v:.5f}" for v in r) + "\n")
    for r in rows:
        if r[0] in (25, 100, 200):
            print(f"  GMV, T = 250, N = {r[0]}, q = {r[1]:.2f}: in-sample / true {r[2]:.4f} (exact (T-N)/(T-1) = {r[5]:.4f}), "
                  f"out-of-sample / true {r[6]:.4f} (1/(1-q) = {1 / (1 - r[1]):.4f})")

    Nf, Tf = 10, 1250
    bet = rng.uniform(0.5, 1.5, Nf); se = rng.uniform(0.15, 0.35, Nf)
    muf = 0.02 + 0.06 * bet; Sig = 0.16**2 * np.outer(bet, bet) + np.diag(se**2)
    def front(mm_, SS, r):
        Si = np.linalg.inv(SS); o = np.ones(len(mm_))
        a, b_, c_ = o @ Si @ o, o @ Si @ mm_, mm_ @ Si @ mm_; dd = a * c_ - b_ * b_
        W = ((c_ - b_ * r) / dd) * (Si @ o)[:, None] + ((a * r - b_) / dd) * (Si @ mm_)[:, None]
        return W, b_ / a
    W, r0 = front(muf, Sig, np.zeros(1))
    rs = r0 + np.linspace(0, 0.45, 181)
    W, _ = front(muf, Sig, rs)
    with open(f"{OUT}/frontiers.dat", "w") as fh:
        fh.write("risk ret\n")
        for k in range(len(rs)): fh.write(f"{np.sqrt(W[:, k] @ Sig @ W[:, k]):.5f} {rs[k]:.5f}\n")
    tg = np.sqrt(W[:, 0] @ Sig @ W[:, 0])
    Lc = np.linalg.cholesky(Sig / Nd)
    fin, fout, gin, gout = [], [], [], []
    sh_true = np.sqrt(muf @ np.linalg.solve(Sig, muf))
    for _ in range(20):
        R = muf / Nd + rng.normal(size=(Tf, Nf)) @ Lc.T
        mh, Sh = Nd * R.mean(0), Nd * np.cov(R.T)
        _, rh0 = front(mh, Sh, np.zeros(1))
        rr = rh0 + np.linspace(0, 0.45, 181)
        Wh, _ = front(mh, Sh, rr)
        fin += [(np.sqrt(Wh[:, k] @ Sh @ Wh[:, k]), rr[k]) for k in range(len(rr))] + [(np.nan, np.nan)]
        fout += [(np.sqrt(Wh[:, k] @ Sig @ Wh[:, k]), muf @ Wh[:, k]) for k in range(len(rr))] + [(np.nan, np.nan)]
        gin.append(np.sqrt(Wh[:, 0] @ Sh @ Wh[:, 0])); gout.append(np.sqrt(Wh[:, 0] @ Sig @ Wh[:, 0]))
    for fn, rws in (("frontiers_in.dat", fin), ("frontiers_out.dat", fout)):
        with open(f"{OUT}/{fn}", "w") as fh:
            fh.write("risk ret\n")
            for a, b_ in rws: fh.write(f"{a:.5f} {b_:.5f}\n")
    k20 = np.searchsorted(rs, r0 + 0.2)
    ins_, outs_ = np.array(fin), np.array(fout)
    print(f"frontiers, N = {Nf}, T = {Tf}: the true least risk {tg:.4f} at the return {r0:.4f}; the in-sample least risks from "
          f"{min(gin):.4f} to {max(gin):.4f}, the same portfolios out of sample from {min(gout):.4f} to {max(gout):.4f}")
    tr20 = np.sqrt(W[:, k20] @ Sig @ W[:, k20])
    print(f"  at the return 0.2 above the least-risk portfolio the true frontier needs the risk {tr20:.4f}; the true "
          f"slope of the tangent through the origin (the largest Sharpe ratio) {sh_true:.4f}")
    rr_out = outs_[~np.isnan(outs_[:, 0])].reshape(20, -1, 2)
    rr_in = ins_[~np.isnan(ins_[:, 0])].reshape(20, -1, 2)
    print(f"  the in-sample frontiers at their own least-risk return + 0.2: risks from {rr_in[:, k20, 0].min():.4f} to "
          f"{rr_in[:, k20, 0].max():.4f}; the same portfolios out of sample: risks from {rr_out[:, k20, 0].min():.4f} to "
          f"{rr_out[:, k20, 0].max():.4f} and returns from {rr_out[:, k20, 1].min():.4f} to {rr_out[:, k20, 1].max():.4f}")

    g0 = np.random.default_rng(0)
    mu0, sg0 = g0.uniform(-0.05, 0.05, n), g0.uniform(-0.2, 0.2, n)
    Er = Nd * (np.exp(mu0 / Nd) - 1); Vv = Nd * np.exp(2 * mu0 / Nd) * (np.exp(sg0**2 / Nd) - 1)
    gt = gvals(Er, np.diag(Vv), Fx); o = np.argsort(gt)
    cnt, cmv, reg, regmv = np.zeros(nf), np.zeros(nf), [], []
    rs_ = np.random.default_rng(1303)
    for _ in range(4000):
        W_ = np.cumsum(rs_.normal(size=(Nd, n)), 0) / np.sqrt(Nd)
        kk_ = np.arange(1, Nd + 1)[:, None]
        Sp = np.vstack([np.ones(n), np.exp((mu0 - sg0**2 / 2) * kk_ / Nd + sg0 * W_)])
        r_ = Sp[1:] / Sp[:-1] - 1
        mh, Ch = Nd * r_.mean(0), Nd * np.cov(r_.T)
        i = int(np.argmin(gvals(mh, Ch, Fx))); cnt[i] += 1; reg.append(gt[i] - gt.min())
        j = int(np.argmin(np.einsum("si,ij,sj->s", Fx, Ch, Fx))); cmv[j] += 1; regmv.append(gt[j] - gt.min())
    jmv = int(np.argmax(cmv))
    with open(f"{OUT}/resample.dat", "w") as fh:
        fh.write("rank dg freq ws mv label\n")
        for r, i in enumerate(o):
            fh.write(f"{r + 1} {gt[i] - gt.min():.5f} {cnt[i] / cnt.sum():.4f} {int(i == iopt)} {int(i == jmv)} {lab(Fx[i])}\n")
    print(f"resampling the worked example from its true model (drifts {', '.join(f'{v:.3f}' for v in mu0)}, volatilities "
          f"{', '.join(f'{abs(v):.3f}' for v in sg0)}): the true optimum {lab(Fx[o[0]])}; the estimated d* = {lab(dstar)} is "
          f"{list(o).index(iopt) + 1}th of {nf} in truth, {gt[iopt] - gt.min():.4f} above")
    print(f"  over 4000 fresh samples of {Nd} days, the exact solution of the estimated problem is the true optimum with the "
          f"frequency {cnt[o[0]] / cnt.sum():.3f}, one of the five best with {cnt[o[:5]].sum() / cnt.sum():.3f}; its mean "
          f"regret {np.mean(reg):.4f}, against {gt.mean() - gt.min():.4f} for a portfolio chosen uniformly at random")
    print(f"  the least estimated variance picks {lab(Fx[jmv])} (the {list(o).index(jmv) + 1}th) in {cmv[jmv] / cmv.sum():.3f} "
          f"of the samples, with the mean regret {np.mean(regmv):.4f}; the true gaps between consecutive portfolios from "
          f"{np.diff(gt[o]).min():.4f} to {np.diff(gt[o]).max():.4f}; the standard deviations of the estimated means "
          f"{', '.join(f'{abs(v):.3f}' for v in sg0)}, against the drifts in [-0.05, 0.05]")

    # ----------------------------------------------------------------------- the solvers on the worked example
    Ss, c, _, _ = solvers(m, C)
    rng = np.random.default_rng(1300)
    table = {}
    for S in Ss:
        res = optimise(S, PMAX, rng, nrand=100)
        pr0 = np.abs(S.psi0)**2
        table[S.name] = [(pr0 @ S.cost - S.F0, pr0[S.iopt], pr0[S.feas].sum())]
        for p, (v, x) in enumerate(res, 1):
            pr = S.probs(x); table[S.name].append((pr @ S.cost - S.F0, pr[S.iopt], pr[S.feas].sum()))
        print(f"  {S.name:9s}: " + "; ".join(f"p={p}: {e:.4f}, {pp:.4f}, {pf:.3f}" for p, (e, pp, pf) in enumerate(table[S.name])))
        if S.name == "cvar":
            print(f"    the CVaR at alpha = {S.obj} of the trained states: " +
                  ", ".join(f"{v - S.F0:.2e}" for v, _ in res))
    with open(f"{OUT}/solvers_depth.dat", "w") as fh:
        fh.write("p " + " ".join(f"{a}e {a}p {a}f" for a in NAMES) + "\n")
        for p in range(PMAX + 1):
            fh.write(f"{p} " + " ".join(f"{max(table[a][p][0], 1e-6):.6f} {table[a][p][1]:.5f} {table[a][p][2]:.5f}"
                                         for a in NAMES) + "\n")
    print(f"solvers on the worked example (F - F*, P(d*), P(budget)), as above; the relaxed optimum c* = "
          f"({', '.join(f'{v:.3f}' for v in c)})")
    # the ring mixer from its own top eigenvector
    Sring = [S for S in Ss if S.name == "ring"][0]
    Sring2 = Solver("ring-top", Sring.cost, top, Sring.V, Sring.s, Sring.per, "E", Sring.iopt, Sring.feas, Sring.F0)
    res = optimise(Sring2, PMAX, np.random.default_rng(1300), nrand=100)
    print("  the ring XY mixer from its top eigenvector: " +
          "; ".join(f"p={p}: {Sring2.probs(x) @ Sring2.cost - Sring2.F0:.4f}, {Sring2.probs(x)[Sring2.iopt]:.4f}"
                    for p, (_, x) in enumerate(res, 1)))

    # ----------------------------------------------------------------------- 100 instances
    with Pool(8) as pool:
        allr = dict(pool.map(run_instance, range(1, 101)))
    seeds = sorted(allr)
    with open(f"{OUT}/solvers_inst.dat", "w") as fh:
        fh.write("p " + " ".join(f"{a}25 {a}50 {a}75" for a in NAMES) + "\n")
        for p in range(1, PMAX + 1):
            vals = []
            for a in NAMES:
                v = [allr[s][a][p - 1][1] for s in seeds]; vals += list(np.percentile(v, [25, 50, 75]))
            fh.write(f"{p} " + " ".join(f"{v:.5f}" for v in vals) + "\n")
    for p in range(1, PMAX + 1):
        print(f"  100 instances, p = {p}: the median P(d*) " + ", ".join(
            f"{a} {np.median([allr[s][a][p - 1][1] for s in seeds]):.3f}" for a in NAMES) + "; the median F - F* " + ", ".join(
            f"{a} {np.median([allr[s][a][p - 1][0] for s in seeds]):.4f}" for a in NAMES))
    for a in NAMES:
        best = sum(1 for s in seeds if allr[s][a][PMAX - 1][1] >= max(allr[s][b][PMAX - 1][1] for b in NAMES) - 1e-12)
        print(f"  {a}: the largest P(d*) at depth {PMAX} on {best} instances; at depth zero the median P(d*) "
              f"{np.median([allr[s][a + '0'][1] for s in seeds]):.4f}")
    print(f"  the classical rounding of the relaxation over P to its {Bud} largest coordinates finds d* on "
          f"{sum(allr[s]['round'] for s in seeds)} of {len(seeds)} instances; the median overlap d*^T c*/B "
          f"{np.median([allr[s]['overlap'] for s in seeds]):.3f}")

if __name__ == "__main__":
    main()

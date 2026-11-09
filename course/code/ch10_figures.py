"""Numerical data for the figures and the numbers quoted in Lecture 10 (rounded warm starts).

Weighted MaxCut on complete graphs with integer weights uniform in {-10, ..., 10}, as in Egger, Marecek and Woerner
(2021): the cut of z in {-1, 1}^n is C(z) = (1/4) sum_{ij} w_ij (1 - z_i z_j), with the sum over ordered pairs. The
worked example is the first instance of the generator on six nodes (seed 0). The Goemans-Williamson relaxation is
solved by cvxpy (SCS), and a random hyperplane with a Gaussian normal r rounds it to z = sign(U r). The warm state for a
cut z puts qubit i at c_i = eps if z_i = +1 and 1 - eps if z_i = -1, with theta_i = 2 arcsin sqrt(c_i); a layer applies
exp(i gamma C) and the mixer exp(i beta n'_i . sigma) on each qubit, a rotation by -2 beta about the axis
n'_i = (-sin theta_i, 0, cos theta_i) (modified, as in the lecture) or n_i = (sin theta_i, 0, cos theta_i) (the
continuous warm start of Lecture 11). Standard QAOA is eps = 1/2 with the continuous mixer. Everything is simulated
exactly.

Writes, into figures/ch10/:
  gw_vectors.dat    i, the two coordinates of the GW vectors of the worked example (the SDP solution has rank two),
                    and the sides of the warm cut
  gw_arcs.dat       the arcs of the directions of the hyperplane normal in that plane, with the cut each produces and
                    its value
  bloch.dat, bloch_pts.dat  the projected Bloch sphere with the paths by which beta = pi/2 maps the two warm states to the poles
  ws10_land.png, std10_land.png  the depth-one landscapes of the expected cut over (beta, gamma), warm and standard
  ws10_depth.dat    p and the best expected cut and the probability of the maximum cut there, warm and standard
  ws10_eps.dat      eps and the quartiles over the pairs of a graph and a GW cut (46: up to five distinct cuts
                    for each of ten graphs) of the depth-one expected cut relative to
                    the maximum cut, with the modified and with the continuous mixer
  rqaoa_steps.dat   the recursion of WS-RQAOA on the worked example: the step, the pair eliminated, its correlation, and
                    the weights of the reduced graph
  rqaoa_cmp.dat     for 40 graphs on ten nodes, the cut of RQAOA, of WS-RQAOA and of the best of ten GW roundings, relative
                    to the maximum cut
  bm_paths.dat      sample paths of the sticky Brownian motion and of the Krivine diffusion in the (W_u, W_v) plane, u.v = -1/2
  bm_corr.dat       u.v and E[sigma_u sigma_v] for the sticky Brownian motion (simulated), the Krivine diffusion (simulated)
                    and (2/pi) arcsin(u.v), with the ratios (1 - E)/(1 - u.v)
and prints the numbers quoted in the text.
"""
import os, itertools
from collections import Counter
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "figures/ch10"
os.makedirs(OUT, exist_ok=True)

def graph(n, seed):
    g = np.random.default_rng(seed); W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n): W[i, j] = W[j, i] = g.integers(-10, 11)
    return W
def strings(n): return 1 - 2 * np.array([[(s >> (n - 1 - q)) & 1 for q in range(n)] for s in range(2**n)])
def cutvals(W, Z): return 0.25 * (W.sum() - np.einsum("ij,si,sj->s", W, Z, Z))
def sdp(W):
    n = len(W); Y = cp.Variable((n, n), PSD=True)
    prob = cp.Problem(cp.Maximize(0.25 * cp.sum(cp.multiply(W, 1 - Y))), [cp.diag(Y) == 1])
    prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
    Yv = (Y.value + Y.value.T) / 2; lam, V = np.linalg.eigh(Yv); lam = np.clip(lam, 0, None)
    return prob.value, lam[::-1], V[:, ::-1] * np.sqrt(lam[::-1])

class Circuit:
    """Depth-p circuits on the graph W, from a product state with the angles theta, with the modified or continuous mixer."""
    def __init__(self, W):
        self.W, self.n = W, len(W)
        self.Z = strings(self.n); self.C = cutvals(W, self.Z); self.Cmax = self.C.max()
        self.imax = np.where(self.C >= self.Cmax - 1e-9)[0]
    def state(self, th, betas, gammas, modified):
        psi = np.ones(1)
        for t in th: psi = np.kron(psi, [np.cos(t / 2), np.sin(t / 2)])
        psi = psi.astype(complex)
        for be, ga in zip(betas, gammas):
            psi = psi * np.exp(1j * ga * self.C)
            P = psi.reshape((2,) * self.n)
            for i, t in enumerate(th):
                s = -np.sin(t) if modified else np.sin(t)
                U = np.cos(be) * np.eye(2) + 1j * np.sin(be) * np.array([[np.cos(t), s], [s, -np.cos(t)]])
                P = np.moveaxis(np.tensordot(U, P, axes=([1], [i])), 0, i)
            psi = P.reshape(-1)
        return psi
    def evaluate(self, th, x, modified):
        p = len(x) // 2; pr = np.abs(self.state(th, x[:p], x[p:], modified))**2
        return pr @ self.C, pr[self.imax].sum(), pr
    def best1(self, th, modified, gmax=0.3, nb=31, ng=41):
        """The largest expected cut at depth one over beta in [0, pi), gamma in [-gmax, gmax]: a grid, and the
        Nelder-Mead method from its three best points."""
        Bg, Gg = np.linspace(0, np.pi, nb, endpoint=False), np.linspace(-gmax, gmax, ng)
        L = np.array([[self.evaluate(th, np.array([b, g]), modified)[0] for g in Gg] for b in Bg])
        best = None
        for ij in np.argsort(-L, axis=None)[:3]:
            i, j = np.unravel_index(ij, L.shape)
            r = minimize(lambda x: -self.evaluate(th, x, modified)[0], np.array([Bg[i], Gg[j]]), method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 1000})
            if best is None or r.fun < best.fun: best = r
        return best.x, -best.fun
def warm_thetas(z, eps): return 2 * np.arcsin(np.sqrt(np.where(z == 1, eps, 1 - eps)))
def gw_cuts(W, U, g, N):
    R = g.normal(size=(N, len(W))); Z = np.sign(R @ U.T); Z[Z == 0] = 1
    return Z * Z[:, :1]                                   # fix the first node, since z and -z are the same cut

# --------------------------------------------------------------------------- the worked example
W6 = graph(6, 0); P6 = Circuit(W6)
val, lam, U = sdp(W6)
rng = np.random.default_rng(1001)
Zr = gw_cuts(W6, U, rng, 1000); vals = cutvals(W6, Zr)
cnt = Counter(tuple(z.astype(int)) for z in Zr)
sub = sorted(((c, k) for k, c in cnt.items() if cutvals(W6, np.array([k]))[0] < P6.Cmax - 1e-9), reverse=True)
zc = np.array(sub[0][1])
print(f"the worked example: six nodes, the weights {[int(W6[i, j]) for i in range(6) for j in range(i + 1, 6)]}; the maximum cut "
      f"{P6.Cmax:.0f}, at {[tuple(P6.Z[s]) for s in P6.imax][:2]}; the SDP value {val:.3f}, the eigenvalues of Y "
      f"{', '.join(f'{x:.3f}' for x in lam)}; 1000 roundings: mean {vals.mean():.2f}, the maximum cut with probability "
      f"{np.mean(vals >= P6.Cmax - 1e-9):.3f}; the most frequent suboptimal cut {tuple(zc)} of value "
      f"{cutvals(W6, zc[None])[0]:.0f}, {sub[0][0]} times")
print(f"  GW mean / SDP = {vals.mean() / val:.3f}, maximum cut / SDP = {P6.Cmax / val:.3f}; distinct cuts and their counts: "
      f"{sorted(((c, cutvals(W6, np.array([k]))[0]) for k, c in cnt.items()), reverse=True)}")
# the vectors lie in a plane; the normal at the angle phi in that plane gives the cut sign(u_i . (cos phi, sin phi))
assert lam[2] < 1e-6
u2 = U[:, :2]
with open(f"{OUT}/gw_vectors.dat", "w") as fh:
    fh.write("i x y side\n")
    for i in range(6): fh.write(f"{i + 1} {u2[i, 0]:.5f} {u2[i, 1]:.5f} {int(zc[i])}\n")
phis = np.linspace(0, np.pi, 3601)
arcs, prev = [], None
for ph in phis:
    z = np.sign(u2 @ np.array([np.cos(ph), np.sin(ph)])); z = z * z[0]; key = tuple(z.astype(int))
    if key != prev: arcs.append([ph, ph, key]); prev = key
    arcs[-1][1] = ph
with open(f"{OUT}/gw_arcs.dat", "w") as fh:
    fh.write("phi0 phi1 value warm max\n")
    for a0, a1, key in arcs:
        v = cutvals(W6, np.array([key]))[0]
        fh.write(f"{a0:.4f} {a1:.4f} {v:.0f} {int(key == tuple(zc))} {int(v >= P6.Cmax - 1e-9)}\n")
        print(f"  the normal between {np.degrees(a0):.1f} and {np.degrees(a1):.1f} degrees: the cut {key}, value {v:.0f}, "
              f"probability {(a1 - a0) / np.pi:.3f}")

# the Bloch sphere: the warm states for z_i = +1 (c = 1/4, theta = pi/3) and z_i = -1 (c = 3/4, theta = 2 pi/3), and the
# rotations by -2 beta, beta from 0 to pi/2, about the modified axes, which end at the opposite poles
az, el = np.radians(30), np.radians(18)
dv = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]); eu = np.array([-np.sin(az), np.cos(az), 0.0])
ev_ = np.cross(dv, eu)
def proj(p): p = np.atleast_2d(p); return p @ eu, p @ ev_, p @ dv
def split(pts):
    u, v, w = proj(pts)
    return np.where(w >= 0, u, np.nan), np.where(w >= 0, v, np.nan), np.where(w < 0, u, np.nan), np.where(w < 0, v, np.nan)
def rot(v, axis, a):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * (axis @ v) * (1 - np.cos(a))
t = np.linspace(0, 2 * np.pi, 361); bs = np.linspace(0, np.pi / 2, 181)
curves = {"eq": np.stack([np.cos(t), np.sin(t), 0 * t], 1), "mer": np.stack([np.sin(t), 0 * t, np.cos(t)], 1)}
for tag, th in (("a", np.pi / 3), ("b", 2 * np.pi / 3)):
    nv, ax_ = np.array([np.sin(th), 0, np.cos(th)]), np.array([-np.sin(th), 0, np.cos(th)])
    curves["path" + tag] = np.array([rot(nv, ax_, -2 * b) for b in bs])
    end = rot(nv, ax_, -np.pi)
    print(f"  Bloch: theta = {th:.4f} (c = {np.sin(th / 2)**2:.2f}): beta = pi/2 rotates ({nv[0]:.3f}, 0, {nv[2]:.3f}) about "
          f"({ax_[0]:.3f}, 0, {ax_[2]:.3f}) to ({end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f})")
    assert np.allclose(end, [0, 0, -np.sign(np.cos(th))], atol=1e-12)
L_ = max(len(c) for c in curves.values())
with open(f"{OUT}/bloch.dat", "w") as fh:
    cols = []
    for k_ in curves: cols += [f"{k_}fu", f"{k_}fv", f"{k_}bu", f"{k_}bv"]
    fh.write(" ".join(cols) + "\n")
    data = [split(curves[k_]) for k_ in curves]
    for j in range(L_):
        fh.write(" ".join("nan" if j >= len(a) or np.isnan(a[j]) else f"{a[j]:.5f}" for d4 in data for a in d4) + "\n")
pts = {"zero": [0, 0, 1], "one": [0, 0, -1], "wa": [np.sin(np.pi / 3), 0, np.cos(np.pi / 3)], "wb": [np.sin(2 * np.pi / 3), 0, np.cos(2 * np.pi / 3)],
       "axa": [-np.sin(np.pi / 3), 0, np.cos(np.pi / 3)], "axb": [-np.sin(2 * np.pi / 3), 0, np.cos(2 * np.pi / 3)]}
with open(f"{OUT}/bloch_pts.dat", "w") as fh:
    fh.write("name u v w\n")
    for k_, p_ in pts.items():
        u, v, w = proj(np.array(p_, float)); fh.write(f"{k_} {u[0]:.5f} {v[0]:.5f} {w[0]:.5f}\n")

# the proposition on the worked example: beta = pi/2, gamma = 0 prepares the cut (the sides exchanged)
th6 = warm_thetas(zc, 0.25)
e_half, pm_half, pr_half = P6.evaluate(th6, np.array([np.pi / 2, 0.0]), True)
s_half = int(np.argmax(pr_half))
assert abs(pr_half.max() - 1) < 1e-12 and np.allclose(P6.Z[s_half], -zc)
e0, pm0, _ = P6.evaluate(th6, np.zeros(2), True)
print(f"  eps = 1/4: at beta = pi/2, gamma = 0 the state is the string {tuple(P6.Z[s_half])} with probability {pr_half.max():.12f}, the "
      f"cut {e_half:.3f}; at zero angles the warm state has the expected cut {e0:.3f}")

# --------------------------------------------------------------------------- landscapes at depth one
cmap = LinearSegmentedColormap.from_list("ylgnbu_r", [np.array(c_) / 255 for c_ in
       [(255, 255, 204), (199, 233, 180), (127, 205, 187), (65, 182, 196), (29, 145, 192), (34, 94, 168), (12, 44, 132)]])
lo_, hi_ = -16.0, 19.0
Bf, Gf = np.linspace(0, np.pi, 241), np.linspace(-0.4, 0.4, 241)
th_std = np.full(6, np.pi / 2)
for tag, th, mod in (("ws10", th6, True), ("std10", th_std, False)):
    L = np.array([[P6.evaluate(th, np.array([b, g]), mod)[0] for b in Bf] for g in Gf])
    plt.imsave(f"{OUT}/{tag}_land.png", np.clip(L, lo_, hi_), cmap=cmap, vmin=lo_, vmax=hi_, origin="lower")
    i, j = np.unravel_index(L.argmax(), L.shape)
    print(f"  landscape {tag}: the expected cut from {L.min():.2f} to {L.max():.3f}, the largest at beta = {Bf[j]:.3f}, "
          f"gamma = {Gf[i]:.3f}")

# --------------------------------------------------------------------------- depth 1 to 5 on the worked example
rng_d = np.random.default_rng(1002)
drows = []
for p in range(1, 6):
    row = [p]
    for nm, th, mod in (("ws", th6, True), ("std", th_std, False)):
        best = None
        for s in range(60):
            x0 = np.concatenate([rng_d.uniform(-np.pi, np.pi, p), rng_d.uniform(-0.3, 0.3, p)])
            if nm == "ws" and s < 20:
                x0 = np.concatenate([[np.pi / 2], np.zeros(p - 1), np.zeros(p)]) + rng_d.normal(scale=0.3, size=2 * p)
            r = minimize(lambda x: -P6.evaluate(th, x, mod)[0], x0, method="L-BFGS-B")
            if best is None or r.fun < best.fun: best = r
        e, pm, _ = P6.evaluate(th, best.x, mod)
        row += [e, pm]
    drows.append(row)
    print(f"  depth {p}: the best expected cut {row[1]:.3f} (warm, the maximum cut with probability {row[2]:.3f}) and "
          f"{row[3]:.3f} (standard, {row[4]:.3f})")
with open(f"{OUT}/ws10_depth.dat", "w") as fh:
    fh.write("p ews pws estd pstd\n")
    for r_ in drows: fh.write(" ".join(f"{x:.5f}" for x in r_) + "\n")

# --------------------------------------------------------------------------- eps on ten-node graphs
rng_e = np.random.default_rng(1003)
epss = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
res = {(e, m): [] for e in epss for m in (True, False)}
gwr = []
for gseed in range(1, 11):
    W = graph(10, gseed); Pg = Circuit(W); _, _, Ug = sdp(W)
    Zg = gw_cuts(W, Ug, rng_e, 200)
    distinct = []
    for z in Zg:
        if not any(np.array_equal(z, d) for d in distinct): distinct.append(z)
        if len(distinct) == 5: break
    for z in distinct:
        gwr.append(cutvals(W, z[None])[0] / Pg.Cmax)
        for e in epss:
            for m in (True, False):
                _, eb = Pg.best1(warm_thetas(z, e), m)
                res[(e, m)].append(eb / Pg.Cmax)
with open(f"{OUT}/ws10_eps.dat", "w") as fh:
    fh.write("eps m25 m50 m75 c25 c50 c75\n")
    for e in epss:
        qm, qc = np.percentile(res[(e, True)], [25, 50, 75]), np.percentile(res[(e, False)], [25, 50, 75])
        fh.write(f"{e:.3f} " + " ".join(f"{x:.5f}" for x in list(qm) + list(qc)) + "\n")
        print(f"  eps = {e}: the median ratio {qm[1]:.4f} with the modified mixer and {qc[1]:.4f} with the continuous one")
print(f"  the {len(gwr)} GW cuts: the median ratio {np.median(gwr):.4f}; with the modified mixer at eps = 1/4 the ratio is at "
      f"least the cut's on {np.mean(np.array(res[(0.25, True)]) >= np.array(gwr) - 1e-6) * 100:.0f}% of the pairs, and above it "
      f"by more than 1e-6 on {int(np.sum(np.array(res[(0.25, True)]) > np.array(gwr) + 1e-6))}, by at most "
      f"{np.max(np.array(res[(0.25, True)]) - np.array(gwr)):.2e}")

# --------------------------------------------------------------------------- recursive QAOA
def reduce_graph(W, i, j, s):
    """Eliminate node i by z_i = s z_j: the weights of i move to j with the sign s, and the edge ij becomes a constant."""
    n = len(W); keep = [k for k in range(n) if k != i]
    W2 = W.copy()
    for k in range(n):
        if k not in (i, j): W2[j, k] += s * W[i, k]; W2[k, j] = W2[j, k]
    const = 0.5 * W[i, j] * (1 - s)
    return W2[np.ix_(keep, keep)], const
def correlations(Pc, pr):
    return np.einsum("s,si,sj->ij", pr, Pc.Z, Pc.Z)
def rqaoa(W, nstop, warm, rng_, log=None):
    """Returns the cut of the original graph found by (WS-)RQAOA with depth one."""
    labels = list(range(len(W))); rel = {}; const = 0.0; Wc = W.copy(); step = 0
    while len(Wc) > nstop:
        Pc = Circuit(Wc)
        if warm:
            _, _, Uc = sdp(Wc); Zc = gw_cuts(Wc, Uc, rng_, 10)
            vals_ = cutvals(Wc, Zc); order = np.argsort(-vals_); chosen = []
            for k in order:
                if not any(np.array_equal(Zc[k], c) for c in chosen): chosen.append(Zc[k])
                if len(chosen) == 5: break
            M = np.zeros_like(Wc)
            for z in chosen:
                th = warm_thetas(z, 0.25); x, _ = Pc.best1(th, True)
                M += correlations(Pc, np.abs(Pc.state(th, x[:1], x[1:], True))**2) / len(chosen)
        else:
            th = np.full(len(Wc), np.pi / 2); x, _ = Pc.best1(th, False)
            M = correlations(Pc, np.abs(Pc.state(th, x[:1], x[1:], False))**2)
        np.fill_diagonal(M, 0)
        i, j = np.unravel_index(np.argmax(np.abs(M)), M.shape)
        s = 1.0 if M[i, j] >= 0 else -1.0
        if log is not None: log.append((step, labels[i], labels[j], M[i, j], Wc.copy(), list(labels)))
        rel[labels[i]] = (labels[j], s)
        Wc, c_ = reduce_graph(Wc, i, j, s); const += c_
        labels = [l for k, l in enumerate(labels) if k != i]; step += 1
    Zs = strings(len(Wc)); vs = cutvals(Wc, Zs); zs = Zs[np.argmax(vs)]
    z = {l: zs[k] for k, l in enumerate(labels)}
    for l in reversed(list(rel.keys())):
        jl, s = rel[l]; z[l] = s * z[jl]
    zfull = np.array([z[k] for k in range(len(W))])
    return cutvals(W, zfull[None])[0], vs.max() + const
log = []
c_ws6, _ = rqaoa(W6, 2, True, np.random.default_rng(1004), log)
with open(f"{OUT}/rqaoa_steps.dat", "w") as fh:
    fh.write("step i j corr labels weights\n")
    for st, i, j, m, Wc, labs in log:
        fh.write(f"{st} {i + 1} {j + 1} {m:.4f} {'-'.join(str(l + 1) for l in labs)} "
                 f"{';'.join(str(int(Wc[a, b])) for a in range(len(Wc)) for b in range(a + 1, len(Wc)))}\n")
        print(f"  WS-RQAOA on the worked example, step {st}: nodes {[l + 1 for l in labs]}, eliminate {i + 1} by z_{i + 1} = "
              f"{'+' if m >= 0 else '-'}z_{j + 1}, correlation {m:.3f}")
print(f"  WS-RQAOA on the worked example finds the cut {c_ws6:.0f} (maximum {P6.Cmax:.0f})")
rng_r = np.random.default_rng(1005)
cmp_rows = []
for gseed in range(101, 141):
    W = graph(10, gseed); Pg = Circuit(W); _, _, Ug = sdp(W)
    c_std, _ = rqaoa(W, 4, False, rng_r); c_ws, _ = rqaoa(W, 4, True, rng_r)
    c_gw = cutvals(W, gw_cuts(W, Ug, rng_r, 10)).max()
    cmp_rows.append((gseed, c_std / Pg.Cmax, c_ws / Pg.Cmax, c_gw / Pg.Cmax))
cmp_rows = np.array(cmp_rows)
with open(f"{OUT}/rqaoa_cmp.dat", "w") as fh:
    fh.write("seed rstd rws rgw\n")
    for r_ in cmp_rows: fh.write(f"{int(r_[0])} {r_[1]:.5f} {r_[2]:.5f} {r_[3]:.5f}\n")
for k, nm in ((1, "RQAOA"), (2, "WS-RQAOA"), (3, "best of ten GW")):
    print(f"  40 graphs on ten nodes, {nm}: the maximum cut on {int(np.sum(cmp_rows[:, k] >= 1 - 1e-9))}, the median ratio "
          f"{np.median(cmp_rows[:, k]):.4f}, the least {cmp_rows[:, k].min():.4f}")

# --------------------------------------------------------------------------- roundings as Brownian motions
rng_b = np.random.default_rng(1006)
def unit_pair(t):
    return np.array([1.0, 0.0]), np.array([t, np.sqrt(max(0.0, 1 - t * t))])
def sticky(t, npaths, dt=2e-3, T=8.0, keep=None):
    u, v = unit_pair(t); Wu = np.zeros(npaths); Wv = np.zeros(npaths); au = np.ones(npaths, bool); av = np.ones(npaths, bool)
    path = []
    for k in range(int(T / dt)):
        dB = rng_b.normal(scale=np.sqrt(dt), size=(npaths, 2))
        Wu = np.where(au, Wu + dB @ u, Wu); Wv = np.where(av, Wv + dB @ v, Wv)
        hu, hv = au & (np.abs(Wu) >= 1), av & (np.abs(Wv) >= 1)
        Wu[hu] = np.sign(Wu[hu]); Wv[hv] = np.sign(Wv[hv]); au &= ~hu; av &= ~hv
        if keep is not None and k % 5 == 0: path.append(np.stack([Wu[:keep], Wv[:keep]], 1))
        if not (au.any() or av.any()): break
    Wu = np.sign(Wu); Wv = np.sign(Wv)
    return Wu * Wv, (np.array(path) if keep is not None else None)
def krivine_paths(t, npaths, ts):
    u, v = unit_pair(t)
    dts = np.diff(np.concatenate([[0], ts]))
    B = np.cumsum(rng_b.normal(size=(npaths, len(ts), 2)) * np.sqrt(dts)[None, :, None], axis=1)
    s = np.sqrt(1 - ts)[None, :]
    return 2 * norm.cdf((B @ u) / s) - 1, 2 * norm.cdf((B @ v) / s) - 1
_, sp = sticky(-0.5, 12, keep=12)
ts = 1 - np.exp(-np.linspace(0, 9, 400))
ku, kv = krivine_paths(-0.5, 12, ts)
with open(f"{OUT}/bm_paths.dat", "w") as fh:
    fh.write(" ".join(f"s{k}u s{k}v" for k in range(12)) + " " + " ".join(f"k{k}u k{k}v" for k in range(12)) + "\n")
    L_ = max(len(sp), len(ts))
    for j in range(L_):
        a = [f"{sp[j, k, 0]:.4f} {sp[j, k, 1]:.4f}" if j < len(sp) else "nan nan" for k in range(12)]
        b = [f"{ku[k, j]:.4f} {kv[k, j]:.4f}" if j < len(ts) else "nan nan" for k in range(12)]
        fh.write(" ".join(a + b) + "\n")
tv = np.linspace(-1, 1, 21)
crow = []
for t_ in tv:
    es, _ = sticky(t_, 6000)
    u, v = unit_pair(t_); G = rng_b.normal(size=(40000, 2)); ek = np.mean(np.sign(G @ u) * np.sign(G @ v))
    ex = 2 / np.pi * np.arcsin(t_)
    crow.append((t_, es.mean(), ek, ex))
crow = np.array(crow)
with open(f"{OUT}/bm_corr.dat", "w") as fh:
    fh.write("t esticky ekrivine eexact rsticky rkrivine rexact\n")
    for t_, es, ek, ex in crow:
        rs = (1 - es) / (1 - t_) if t_ < 1 else np.nan; rk = (1 - ek) / (1 - t_) if t_ < 1 else np.nan
        rx = (1 - ex) / (1 - t_) if t_ < 1 else np.nan
        fh.write(f"{t_:.3f} {es:.5f} {ek:.5f} {ex:.5f} " + " ".join("nan" if np.isnan(x) else f"{x:.5f}" for x in (rs, rk, rx)) + "\n")
m = crow[:, 0] < 0.99
rs = (1 - crow[m, 1]) / (1 - crow[m, 0]); rx = (1 - crow[m, 3]) / (1 - crow[m, 0])
tt = np.linspace(-1, 0.999, 20001); gwc = (1 - 2 / np.pi * np.arcsin(tt)) / (1 - tt)
print(f"rounding correlations: the Krivine diffusion agrees with (2/pi) arcsin(t) to {np.abs(crow[:, 2] - crow[:, 3]).max():.4f}; "
      f"the sticky Brownian motion gives E = {crow[5, 1]:.3f} at t = {crow[5, 0]:.1f} against {crow[5, 3]:.3f}; on the grid, "
      f"the least ratio (1 - E)/(1 - t) is {rs.min():.4f} (sticky, at t = {crow[m, 0][np.argmin(rs)]:.1f}) and {rx.min():.4f} "
      f"(exact); alpha_GW = {gwc.min():.5f} at t = {tt[np.argmin(gwc)]:.4f}")

"""Numerical data for the figures and the numbers quoted in Lecture 12 (variational linear solvers).

The linear systems are the Ising-inspired family of Bravo-Prieto et al. (2023): on n qubits,
    A = (sum_j X_j + J sum_j Z_j Z_{j+1} + eta 1) / zeta,   J = 0.1,
with eta and zeta chosen so that the eigenvalues of A fill [1/kappa, 1], and |b> = H^{(x)n}|0> = |+>^n. The ansatz
has layers of R_Y rotations on all qubits, separated by CZ gates on the pairs (0,1), (2,3), ... and (1,2), (3,4), ...
alternately, and a final layer of rotations. Everything is simulated exactly, with states as arrays of 2^n amplitudes.

Writes, into figures/ch12/:
  vqls_train.dat    the iteration of BFGS on the local cost for the worked example (n = 3, kappa = 10, three layers),
                    C_L, C_G, the error D = sqrt(1 - |<x|psi>|^2), and the bounds kappa sqrt(C_G) and kappa sqrt(n C_L)
  vqls_cert.dat     sqrt(C_G) and D for states of the ansatz at random angles (kind 0), along the training (kind 1),
                    and for |x> perturbed along the eigenvectors of the least (kind 2) and the largest (kind 3)
                    eigenvalue of A
  vqls_plateau.dat  n and the variances of the derivatives of C_G and C_L with respect to the first angle, at random
                    angles, with four layers
  vqls_slice.dat    an angle and C_G and C_L along it, at random other angles, for n = 4 and n = 12
  vqls_kappa.dat    kappa and the iterations of conjugate gradients until C_G <= 1e-8 for |b> = |+>^10 and for a
                    random |b>, the bound of conjugate gradients, and the iterations of VQLS (BFGS, n = 4, six layers)
                    until C_G <= 1e-6, the median and the quartiles of five starts
and prints the numbers quoted in the text.
"""
import os
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import cg, LinearOperator, eigsh

OUT = "figures/ch12"
os.makedirs(OUT, exist_ok=True)

def zsign(n, j):
    s = np.ones((2,) * n); idx = [slice(None)] * n; idx[j] = 1; s[tuple(idx)] = -1
    return s.reshape(-1)

class System:
    """The linear system of the family on n qubits with condition number kappa."""
    def __init__(self, n, kappa, J=0.1, b=None):
        self.n, self.J = n, J
        self.zz = [zsign(n, j) * zsign(n, j + 1) for j in range(n - 1)]
        self.bits = np.array([[(s >> (n - 1 - q)) & 1 for q in range(n)] for s in range(2**n)])
        self.kappa = kappa
        self.b = np.ones(2**n) / np.sqrt(2**n) if b is None else b
        if n <= 8:                     # dense, with the eigenvectors and the solution
            M = np.array([self.H(e) for e in np.eye(2**n)]).T
            ev, V = np.linalg.eigh(M)
            self.lmin, self.lmax = ev[0], ev[-1]
        else:                          # the extreme eigenvalues only
            Hop = LinearOperator((2**n, 2**n), matvec=self.H, dtype=float)
            self.lmin = eigsh(Hop, k=1, which="SA", tol=1e-12)[0][0]; self.lmax = eigsh(Hop, k=1, which="LA", tol=1e-12)[0][0]
        self.eta = (self.lmax - kappa * self.lmin) / (kappa - 1)
        self.zeta = self.lmax + self.eta
        if n <= 8:
            self.Amat = (M + self.eta * np.eye(2**n)) / self.zeta
            self.evA, self.VA = (ev + self.eta) / self.zeta, V
            x = np.linalg.solve(self.Amat, self.b); self.x = x / np.linalg.norm(x)
    def H(self, psi):
        P = psi.reshape((2,) * self.n)
        out = sum(np.flip(P, axis=j).reshape(-1) for j in range(self.n))
        return out + self.J * sum(z * psi for z in self.zz)
    def A(self, psi): return (self.H(psi) + self.eta * psi) / self.zeta
    def hadamards(self, v):
        P = v.reshape((2,) * self.n)
        for j in range(self.n):
            P = np.moveaxis(P, j, 0); P = np.stack([P[0] + P[1], P[0] - P[1]]) / np.sqrt(2); P = np.moveaxis(P, 0, j)
        return P.reshape(-1)
    def ansatz(self, th, layers):
        n = self.n; psi = np.zeros(2**n); psi[0] = 1; k = 0
        for l in range(layers + 1):
            P = psi.reshape((2,) * n)
            for q in range(n):
                c, s = np.cos(th[k] / 2), np.sin(th[k] / 2); k += 1
                P = np.moveaxis(P, q, 0); P = np.stack([c * P[0] - s * P[1], s * P[0] + c * P[1]]); P = np.moveaxis(P, 0, q)
            psi = P.reshape(-1)
            if l < layers:
                pairs = [(q, q + 1) for q in range(l % 2, n - 1, 2)]
                for a, c_ in pairs: psi = psi * np.where(self.bits[:, a] & self.bits[:, c_], -1, 1)
        return psi
    def costs(self, psi):
        """C_G = 1 - |<b|phi>|^2 / <phi|phi>, and C_L with the projectors |0><0|_j after undoing U_b = H^n."""
        phi = self.A(psi); nphi = phi @ phi
        CG = 1 - (self.b @ phi)**2 / nphi
        chi = self.hadamards(phi) / np.sqrt(nphi)
        p0 = np.array([np.sum(chi[self.bits[:, j] == 0]**2) for j in range(self.n)])
        return CG, 1 - p0.mean()
    def error(self, psi): return np.sqrt(max(0.0, 1 - (self.x @ psi)**2 / (psi @ psi)))

# --------------------------------------------------------------------------- the worked example
n0, kap0, lay0 = 3, 10.0, 3
S = System(n0, kap0)
assert np.allclose(S.Amat @ S.x / np.linalg.norm(S.Amat @ S.x), S.b) and abs(np.linalg.cond(S.Amat) - kap0) < 1e-9
assert np.allclose([S.A(e) for e in np.eye(2**n0)], S.Amat.T)
print(f"the worked example: n = {n0}, J = 0.1, kappa = {kap0}: the Ising part has the eigenvalues from {S.lmin:.4f} to "
      f"{S.lmax:.4f}; eta = {S.eta:.4f}, zeta = {S.zeta:.4f}; the eigenvalues of A are "
      f"{', '.join(f'{x:.4f}' for x in S.evA)}; |x> = ({', '.join(f'{x:.4f}' for x in S.x)}); "
      f"the coefficients 1/zeta = {1 / S.zeta:.4f} (X_j), J/zeta = {0.1 / S.zeta:.4f} (Z_j Z_j+1), eta/zeta = {S.eta / S.zeta:.4f} (1)")
print(f"  <b|x> = {S.b @ S.x:.4f}, so that D(b, x) = {np.sqrt(1 - (S.b @ S.x)**2):.4f}: |x> is close to |b>, the top "
      f"eigenvector of sum_j X_j, which the couplings J perturb")
L_terms = n0 + (n0 - 1) + 1
print(f"  A has L = {L_terms} unitary terms: <phi|phi> needs {L_terms * (L_terms + 1) // 2} Hadamard tests (the real parts of "
      f"<psi|A_l A_l'|psi>, l <= l'), and the local cost {L_terms * (L_terms + 1) // 2} for each of the n = {n0} projectors")
# the bounds hold on random states
g = np.random.default_rng(1201)
for _ in range(2000):
    v = g.normal(size=2**n0); v /= np.linalg.norm(v)
    CG, CL = S.costs(v); D = S.error(v)
    assert CL <= CG + 1e-12 and CG <= n0 * CL + 1e-12 and np.sqrt(CG) / kap0 <= D + 1e-12 and D <= kap0 * np.sqrt(CG) + 1e-12

# training: BFGS on C_L from random angles, recorded after each iteration
npar = n0 * (lay0 + 1)
th0 = g.uniform(0, 2 * np.pi, npar)
rec = []
def record(th):
    psi = S.ansatz(th, lay0); CG, CL = S.costs(psi); rec.append((CL, CG, S.error(psi)))
record(th0)
res = minimize(lambda th: S.costs(S.ansatz(th, lay0))[1], th0, method="BFGS", callback=record, options={"gtol": 1e-11, "maxiter": 400})
rec = np.array(rec)
with open(f"{OUT}/vqls_train.dat", "w") as fh:
    fh.write("it CL CG D bG bL\n")
    for k, (CL, CG, D) in enumerate(rec):
        fh.write(f"{k} {CL:.6e} {CG:.6e} {D:.6e} {kap0 * np.sqrt(CG):.6e} {kap0 * np.sqrt(n0 * CL):.6e}\n")
ratio = rec[:, 2] / (kap0 * np.sqrt(rec[:, 1]))
k3 = int(np.argmax(rec[:, 0] <= 1e-6))
print(f"training, {npar} angles, {len(rec) - 1} iterations of BFGS: C_L from {rec[0, 0]:.4f} to {rec[-1, 0]:.2e}, C_G from "
      f"{rec[0, 1]:.4f} to {rec[-1, 1]:.2e}, the error D from {rec[0, 2]:.4f} to {rec[-1, 2]:.2e}; C_L <= 1e-6 after {k3} "
      f"iterations, where D = {rec[k3, 2]:.2e} against kappa sqrt(C_G) = {kap0 * np.sqrt(rec[k3, 1]):.2e} and "
      f"kappa sqrt(n C_L) = {kap0 * np.sqrt(n0 * rec[k3, 0]):.2e}; D / (kappa sqrt C_G) is {ratio[0]:.3f} at the start and "
      f"{ratio[-1]:.3f} at the end")

# the certificate: states at random angles, the training, and |x> perturbed along the extreme eigenvectors
cert = []
for _ in range(600):
    psi = S.ansatz(g.uniform(0, 2 * np.pi, npar), lay0); CG, _ = S.costs(psi); cert.append((0, np.sqrt(CG), S.error(psi)))
for CL, CG, D in rec: cert.append((1, np.sqrt(CG), D))
vmin, vmax = S.VA[:, 0], S.VA[:, -1]
for kind, v in ((2, vmin), (3, vmax)):
    v = v - (v @ S.x) * S.x; v /= np.linalg.norm(v)
    for t in np.exp(np.linspace(np.log(1e-5), np.log(3), 60)):
        psi = S.x + t * v; psi /= np.linalg.norm(psi); CG, _ = S.costs(psi); cert.append((kind, np.sqrt(CG), S.error(psi)))
cert = np.array(cert)
with open(f"{OUT}/vqls_cert.dat", "w") as fh:
    fh.write("kind sCG D\n")
    for r in cert: fh.write(f"{int(r[0])} {r[1]:.6e} {r[2]:.6e}\n")
for kind, nm in ((0, "random angles"), (2, "along v_min"), (3, "along v_max")):
    c_ = cert[(cert[:, 0] == kind) & (cert[:, 1] > 1e-12)]
    rr = c_[:, 2] / c_[:, 1]
    print(f"  certificate, {nm}: D / sqrt(C_G) between {rr.min():.3f} and {rr.max():.3f} (the band is [1/kappa, kappa] = "
          f"[{1 / kap0:.1f}, {kap0:.0f}])")

# --------------------------------------------------------------------------- plateaus
rng_p = np.random.default_rng(1202)
lay_p = 4
rows = []
for n in range(2, 13):
    Sn = System(n, 10.0); npn = n * (lay_p + 1)
    dG, dL = [], []
    for _ in range(400):
        th = rng_p.uniform(0, 2 * np.pi, npn); e = np.zeros(npn); e[0] = 1e-5
        cp, cm = Sn.costs(Sn.ansatz(th + e, lay_p)), Sn.costs(Sn.ansatz(th - e, lay_p))
        dG.append((cp[0] - cm[0]) / 2e-5); dL.append((cp[1] - cm[1]) / 2e-5)
    rows.append((n, np.var(dG), np.var(dL)))
    print(f"  plateaus, n = {n}: Var dC_G = {np.var(dG):.3e}, Var dC_L = {np.var(dL):.3e}")
rows = np.array(rows)
with open(f"{OUT}/vqls_plateau.dat", "w") as fh:
    fh.write("n vG vL\n")
    for r in rows: fh.write(f"{int(r[0])} {r[1]:.6e} {r[2]:.6e}\n")
sG = np.polyfit(rows[:, 0], np.log2(rows[:, 1]), 1)[0]; sL = np.polyfit(np.log2(rows[:, 0]), np.log2(rows[:, 2]), 1)[0]
print(f"  fits: Var dC_G ~ 2^({sG:.2f} n); Var dC_L ~ n^({sL:.2f}); from n = 2 to 12 the first falls by "
      f"{rows[0, 1] / rows[-1, 1]:.0f} times, the second by {rows[0, 2] / rows[-1, 2]:.0f}")
# slices along the first angle
ts = np.linspace(0, 2 * np.pi, 121)
sl = {}
for n in (4, 12):
    Sn = System(n, 10.0); npn = n * (lay_p + 1); th = rng_p.uniform(0, 2 * np.pi, npn)
    vals = []
    for t in ts:
        th[0] = t; vals.append(Sn.costs(Sn.ansatz(th, lay_p)))
    sl[n] = np.array(vals)
    print(f"  slice, n = {n}: C_G between {sl[n][:, 0].min():.4f} and {sl[n][:, 0].max():.4f}, C_L between "
          f"{sl[n][:, 1].min():.4f} and {sl[n][:, 1].max():.4f}")
with open(f"{OUT}/vqls_slice.dat", "w") as fh:
    fh.write("t G4 L4 G12 L12\n")
    for j, t in enumerate(ts):
        fh.write(f"{t:.4f} {sl[4][j, 0]:.6f} {sl[4][j, 1]:.6f} {sl[12][j, 0]:.6f} {sl[12][j, 1]:.6f}\n")

# --------------------------------------------------------------------------- the classical baseline
rng_k = np.random.default_rng(1203)
kappas = [2, 5, 10, 20, 50, 100, 200, 500]
rb = rng_k.normal(size=2**10); rb /= np.linalg.norm(rb)
def cg_iters(Sn, b):
    N = 2**Sn.n; its = []
    cg(LinearOperator((N, N), matvec=Sn.A, dtype=float), b, tol=1e-13, maxiter=5000, callback=lambda xk: its.append(xk.copy()))
    def cost(v):
        phi = Sn.A(v / np.linalg.norm(v)); return 1 - (b @ phi)**2 / (phi @ phi)
    return next((i + 1 for i, v in enumerate(its) if cost(v) <= 1e-8), None)
krows = []
for kap in kappas:
    Sp, Sr = System(10, float(kap)), System(10, float(kap), b=rb)
    kp, kr = cg_iters(Sp, Sp.b), cg_iters(Sr, rb)
    bound = 0.5 * np.sqrt(kap) * np.log(2 / 1e-4)
    vq = [np.nan, np.nan, np.nan]
    if kap <= 100:
        S4 = System(4, float(kap)); lay4 = 6; np4 = 4 * (lay4 + 1); its = []
        for _ in range(5):
            cnt, first = [0], [None]
            def cb(th):
                cnt[0] += 1
                if first[0] is None and S4.costs(S4.ansatz(th, lay4))[0] <= 1e-6: first[0] = cnt[0]
            minimize(lambda th: S4.costs(S4.ansatz(th, lay4))[1], rng_k.uniform(0, 2 * np.pi, np4), method="BFGS",
                     callback=cb, options={"gtol": 1e-12, "maxiter": 3000})
            its.append(first[0] if first[0] is not None else np.nan)
        vq = list(np.nanpercentile(its, [25, 50, 75]))
        print(f"  kappa = {kap}: VQLS (n = 4, six layers) reaches C_G <= 1e-6 after {its} iterations of BFGS")
    krows.append((kap, kp, kr, bound, *vq))
    print(f"  kappa = {kap}: conjugate gradients until C_G <= 1e-8: {kp} iterations for |+>^10, {kr} for a random |b>; "
          f"the bound (1/2) sqrt(kappa) ln(2/1e-4) = {bound:.1f}")
with open(f"{OUT}/vqls_kappa.dat", "w") as fh:
    fh.write("kappa cgplus cgrand bound vq25 vq50 vq75\n")
    for r in krows: fh.write(" ".join("nan" if (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}" for x in r) + "\n")
eps_c = 0.01
print(f"certifying D <= {eps_c} needs C_G <= (eps/kappa)^2: {(eps_c / 10)**2:.0e} at kappa = 10 and {(eps_c / 100)**2:.0e} at "
      f"kappa = 100; by Hoeffding's inequality, an additive precision delta takes of the order of 1/delta^2 copies, "
      f"{1 / ((eps_c / 10)**2)**2:.0e} and {1 / ((eps_c / 100)**2)**2:.0e}")

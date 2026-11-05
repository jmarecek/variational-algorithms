"""Numerical data for the figures and the numbers quoted in Lecture 7 (the parameter-shift rule).

The example is the worked example of Lecture 5, depth-one QAOA for MaxCut on the 4-cycle, simulated
gate by gate in the conventions of code/ch05_worked_example.py, at the starting point
(beta, gamma) = (0.1, -0.1). The single-gate function f(phi) is the energy as a function of the angle
phi of the R_Z gate of edge 12, with all the other angles fixed; at the starting point phi = -0.1.

Writes, into figures/ch07/:
  trig_shared.dat   x and, along a parameter shared by three Pauli rotations, the cost f (degree 3), the degree-2
                    interpolant f2 from 5 values, the reconstruction fhat from 7 estimates, and f -+ sqrt(7) eta
  trig_band.dat     the band f -+ sqrt(7) eta as a closed polygon
  trig_samples7.dat, trig_samples5.dat   the 7 and 5 equally spaced points, the values and the estimates
  step_rmse.dat     h and the root-mean-square error of the central difference with step h (fd...)
                    and of the shift rule with shift h (sr...), for f'(-0.1), with N = 2e3, 2e4 and
                    2e5 measured copies in all, divided equally between the two circuits
  step_sampled.dat  h and the root-mean-square error of 4000 simulated estimates of each kind, N = 2e3
  copies_rmse.dat   N, the error of the central difference at its best step, that step, and the
                    error of the shift rule with shift pi/2
and prints the numbers quoted in the text, after checking every formula of the lecture numerically.
"""
import os
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar

OUT = "figures/ch07"
os.makedirs(OUT, exist_ok=True)
n, edges = 4, [(0, 1), (1, 2), (2, 3), (3, 0)]
rng = np.random.default_rng(2027)

# --------------------------------------------------------------------------- the circuit of Lecture 5
Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
def rz(t): return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
def rx(t): return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])

def apply_1q(psi, U, q):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)

def apply_cnot(psi, c, t):
    psi = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[c] = 1
    sub = psi[tuple(idx)]
    t2 = t if t < c else t - 1
    psi[tuple(idx)] = np.flip(sub, axis=t2)
    return psi.reshape(-1)

def circuit(theta_zz, theta_x):
    psi = np.zeros(2**n, complex); psi[0] = 1
    for q in range(n):
        psi = apply_1q(psi, Hd, q)
    for (i, j), t in zip(edges, theta_zz):
        psi = apply_cnot(psi, i, j)
        psi = apply_1q(psi, rz(t), j)
        psi = apply_cnot(psi, i, j)
    for q, t in enumerate(theta_x):
        psi = apply_1q(psi, rx(t), q)
    return psi

bits = np.array([[(x >> (n - 1 - q)) & 1 for q in range(n)] for x in range(2**n)])
cuts = np.array([sum(b[i] != b[j] for i, j in edges) for b in bits])
vals = -cuts.astype(float)                       # the measured value of H_C for each bit string

def moments(theta_zz, theta_x):
    """The energy and the variance of a single measured value."""
    p = np.abs(circuit(theta_zz, theta_x))**2
    m = p @ vals
    return m, p @ vals**2 - m**2

def angles(beta, gamma): return np.full(len(edges), gamma), np.full(n, 2 * beta)
def F(beta, gamma): return moments(*angles(beta, gamma))[0]
def F_closed(beta, gamma): return -2 + np.sin(4 * beta) * np.sin(2 * gamma)
for b, g in [(0.1, 0.3), (0.7, -0.4)]:
    assert abs(F(b, g) - F_closed(b, g)) < 1e-12

beta0, gamma0 = 0.1, -0.1
tz0, tx0 = angles(beta0, gamma0)
def single(phi):
    """The energy and the variance as functions of the angle of the R_Z gate of edge 12."""
    tz = tz0.copy(); tz[0] = phi
    return moments(tz, tx0)
def f(phi): return single(phi)[0]
def var(phi): return single(phi)[1]

def deriv(g, x, h=1e-4):
    """A derivative of a smooth function to about ten digits (fourth-order central difference)."""
    return (-g(x + 2 * h) + 8 * g(x + h) - 8 * g(x - h) + g(x - 2 * h)) / (12 * h)

# --------------------------------------------------------------------------- the frequencies of one gate
# the generator of gamma is H_C, with the eigenvalues -cut(x); as a function of gamma, F has the frequencies 0, 2, 4
mult = {int(-v): int(np.sum(vals == v)) for v in sorted(set(vals), reverse=True)}
assert mult == {0: 2, 2: 12, 4: 2}
xs_ = 2 * np.pi * np.arange(5) / 5                        # in x = 2 gamma, R = 2: five values determine f
fx = np.array([F(beta0, x_ / 2) for x_ in xs_])
a_ = [fx.mean()] + [2 * np.mean(fx * np.cos(l * xs_)) for l in (1, 2)]
b_ = [2 * np.mean(fx * np.sin(l * xs_)) for l in (1, 2)]
assert abs(a_[0] + 2) < 1e-12 and max(abs(a_[1]), abs(a_[2]), abs(b_[1])) < 1e-12 and abs(b_[0] - np.sin(4 * beta0)) < 1e-12
print(f"frequencies of gamma: eigenvalues of H_C (minus the cut) with multiplicities {mult}; at beta = {beta0}, "
      f"f = {a_[0]:.4f} + {b_[0]:.4f} sin 2 gamma, the coefficients of frequency 4 being zero")

# --------------------------------------------------------------------------- the rule for one gate
phi0 = gamma0
fp, fm, f0 = f(phi0 + np.pi / 2), f(phi0 - np.pi / 2), f(phi0)
d_true = deriv(f, phi0)
print(f"single gate: f(phi0) = {f0:.4f}, f(phi0 + pi/2) = {fp:.4f}, f(phi0 - pi/2) = {fm:.4f}")
print(f"  shift rule (f+ - f-)/2 = {(fp - fm) / 2:.6f}, derivative {d_true:.6f}")
assert abs((fp - fm) / 2 - d_true) < 1e-8
for s in (0.3, 1.0, 2.5):
    assert abs((f(phi0 + s) - f(phi0 - s)) / (2 * np.sin(s)) - d_true) < 1e-8
a = (fp + fm) / 2                                # f = a + b cos phi + c sin phi
M2 = np.array([[np.cos(phi0), np.sin(phi0)], [np.cos(phi0 + np.pi / 2), np.sin(phi0 + np.pi / 2)]])
b, c = np.linalg.solve(M2, [f0 - a, fp - a])
for t in np.linspace(-3, 3, 7):
    assert abs(a + b * np.cos(t) + c * np.sin(t) - f(t)) < 1e-12
amp, ph = np.hypot(b, c), np.arctan2(c, b)        # f = a + amp cos(phi - ph)
print(f"  f(phi) = {a:.5f} + {b:.5f} cos phi + {c:.5f} sin phi = {a:.5f} + {amp:.5f} cos(phi - {ph:.5f})")
print(f"  variances of one measured value: at phi0 {var(phi0):.4f}, at phi0 +- pi/2 {var(phi0 + np.pi/2):.4f}, {var(phi0 - np.pi/2):.4f}")
print(f"  Lecture 5 estimates with 1000 copies: f+ = -1.832, f- = -2.260, giving {(-1.832 + 2.260) / 2:.3f}")

# the commutator form: f' = i <[A, O]> with A = Z/2 on qubit 2, in the Heisenberg picture
# (checked through the matrices: i[P, O] = R(pi/2)^+ O R(pi/2) - R(-pi/2)^+ O R(-pi/2) for a Pauli P)
Pm = np.kron(np.kron(np.eye(2), np.diag([1, -1])), np.eye(4)).astype(complex)
Om = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16)); Om = Om + Om.conj().T
Rp, Rm = expm(-1j * np.pi / 4 * Pm), expm(1j * np.pi / 4 * Pm)
assert np.allclose(1j * (Pm @ Om - Om @ Pm), Rp.conj().T @ Om @ Rp - Rm.conj().T @ Om @ Rm)

# Rotosolve: the minimiser along the coordinate from the same three values
phistar = phi0 + np.pi - np.arctan2(fm - fp, 2 * f0 - fp - fm)
grid = np.linspace(-np.pi, np.pi, 200001)
fg = a + b * np.cos(grid) + c * np.sin(grid)
assert abs(np.angle(np.exp(1j * (phistar - grid[np.argmin(fg)])))) < 1e-4
print(f"  minimiser along the coordinate: phi* = {np.angle(np.exp(1j*phistar)):.4f}, f(phi*) = {f(phistar):.4f} = a - amp = {a - amp:.4f}")

# second derivatives: two gates (R_Z of edges 12 and 23) and the diagonal formula
def f2(p1, p2):
    tz = tz0.copy(); tz[0], tz[1] = p1, p2
    return moments(tz, tx0)[0]
q = np.pi / 2
mixed = (f2(phi0 + q, phi0 + q) - f2(phi0 + q, phi0 - q) - f2(phi0 - q, phi0 + q) + f2(phi0 - q, phi0 - q)) / 4
mixed_true = deriv(lambda y: deriv(lambda x: f2(x, y), phi0, 1e-3), phi0, 1e-3)
diag = (f(phi0 + np.pi) - f(phi0)) / 2
diag_true = deriv(lambda x: deriv(f, x, 1e-3), phi0, 1e-3)
print(f"  second derivatives: mixed {mixed:.6f} ({mixed_true:.6f}), diagonal {diag:.6f} ({diag_true:.6f})")
assert abs(mixed - mixed_true) < 1e-6 and abs(diag - diag_true) < 1e-6

# --------------------------------------------------------------------------- the general rule
def weights(R):
    """Shifts x_mu = (2 mu - 1) pi / (2R) and weights y_mu, mu = 1..R, of the equidistant rule."""
    mu = np.arange(1, R + 1)
    x = (2 * mu - 1) * np.pi / (2 * R)
    return x, (-1.0)**(mu - 1) / (4 * R * np.sin(x / 2)**2)

for R in range(1, 9):                             # exact on cos(l x), sin(l x), l <= R; l1-norm R
    x, y = weights(R)
    for l in range(R + 1):
        assert abs(np.sum(y * (np.sin(l * x) - np.sin(-l * x))) - l) < 1e-10
        assert abs(np.sum(y * (np.cos(l * x) - np.cos(-l * x)))) < 1e-12
    assert abs(2 * np.sum(np.abs(y)) - R) < 1e-10
for R in (1, 2, 3):
    x, y = weights(R)
    print(f"  R = {R}: shifts {np.round(x, 4)}, weights {np.round(y, 4)}, l1-norm {2*np.sum(np.abs(y)):.4f}")
x3, y3 = weights(3)
print("  R = 3, all 2R points: x =", np.round(np.degrees(np.r_[x3, 2*np.pi - x3[::-1]]), 1),
      "weights", np.round(np.r_[y3, -y3[::-1]], 4))

# the worked example: gamma enters through e^{-i gamma H_C}, whose eigenvalues 0, -2, -4 give the
# frequencies 2 and 4, so R = 2 in x = 2 gamma; beta through e^{-i beta H_M}, frequencies 2, 4, 6, 8,
# so R = 4 in x = 2 beta
gt_b, gt_g = 4 * np.cos(4 * beta0) * np.sin(2 * gamma0), 2 * np.sin(4 * beta0) * np.cos(2 * gamma0)
x, y = weights(2)
g_gamma = 2 * sum(yy * (F(beta0, gamma0 + xx / 2) - F(beta0, gamma0 - xx / 2)) for xx, yy in zip(x, y))
x4, y4 = weights(4)
g_beta = 2 * sum(yy * (F(beta0 + xx / 2, gamma0) - F(beta0 - xx / 2, gamma0)) for xx, yy in zip(x4, y4))
print(f"worked example: dF/dgamma by the rule with R = 2 (4 circuits): {g_gamma:.6f}, exact {gt_g:.6f}")
print(f"                dF/dbeta  by the rule with R = 4 (8 circuits): {g_beta:.6f}, exact {gt_b:.6f}")
print(f"                gamma shifts {np.round(x/2, 4)} = pi/8, 3pi/8; weights 2*y = {np.round(2*y, 4)}")
assert abs(g_gamma - gt_g) < 1e-10 and abs(g_beta - gt_b) < 1e-10

# standard deviations with M = 1000 copies for each circuit, and per copy with the copies allocated
# in proportion to |weight| * sigma
M = 1000
gate_terms = []
for e in range(4):
    for sg in (1, -1):
        tz = tz0.copy(); tz[e] += sg * np.pi / 2
        gate_terms.append((0.5, moments(tz, tx0)[1]))
terms = [(2 * yy, moments(*angles(beta0, gamma0 + sg * xx / 2))[1]) for xx, yy in zip(x, y) for sg in (1, -1)]
sd_gate = np.sqrt(sum(w**2 * v for (w, v) in gate_terms) / M)
sd_gen = np.sqrt(sum(w**2 * v for (w, v) in terms) / M)
ell_gate = sum(abs(w) * np.sqrt(v) for (w, v) in gate_terms)
ell = sum(abs(w) * np.sqrt(v) for (w, v) in terms)
print(f"  dF/dgamma with M = {M} copies per circuit: gate by gate (8 circuits) sd {sd_gate:.4f}; "
      f"general rule (4 circuits) sd {sd_gen:.4f}")
print(f"  per copy, with optimal allocation: N * variance = {ell_gate**2:.3f} (gate by gate), {ell**2:.3f} (general rule)")

# --------------------------------------------------------------------------- the stochastic rule
d2 = 4
A = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2)) / 2            # X on the first qubit, halved
B = rng.normal(size=(d2, d2)) + 1j * rng.normal(size=(d2, d2)); B = (B + B.conj().T) / 2
O2 = rng.normal(size=(d2, d2)) + 1j * rng.normal(size=(d2, d2)); O2 = O2 + O2.conj().T
psi2 = rng.normal(size=d2) + 1j * rng.normal(size=d2); psi2 /= np.linalg.norm(psi2)
def g_noncomm(th):
    u = expm(-1j * (th * A + B)) @ psi2
    return np.real(np.vdot(u, O2 @ u))
th = 0.7
Gm = th * A + B
def split(s, sign):
    R_ = expm(-1j * sign * np.pi / 2 * A)
    u = expm(-1j * s * Gm) @ R_ @ expm(-1j * (1 - s) * Gm) @ psi2
    return np.real(np.vdot(u, O2 @ u))
ss, ww = np.polynomial.legendre.leggauss(40)
ss, ww = (ss + 1) / 2, ww / 2
stoch = 0.5 * sum(w * (split(s, 1) - split(s, -1)) for s, w in zip(ss, ww))
print(f"stochastic rule on two qubits: {stoch:.8f}, derivative {deriv(g_noncomm, th):.8f}, "
      f"plain shift rule {(g_noncomm(th + np.pi/2) - g_noncomm(th - np.pi/2))/2:.8f}")
assert abs(stoch - deriv(g_noncomm, th)) < 1e-7

# --------------------------------------------------------------------------- the cost as a trigonometric polynomial
# Sakos, Varvitsiotis, Korpas and Lin: with integer spectral differences, the cost is a trigonometric polynomial
# whose degree in a parameter is at most the sum of the spectral widths of the gates that share it; its values
# on a grid of prod (2 R_j + 1) points determine it, and the sum-of-squares hierarchy minimises it
import cvxpy as cp
rgt = np.random.default_rng(2029)            # a generator of its own, so that the later samples do not change

def grid_axes(Rs): return [2 * np.pi * np.arange(2 * R + 1) / (2 * R + 1) for R in Rs]
def on_grid(func, Rs):
    pts = np.array(np.meshgrid(*grid_axes(Rs), indexing="ij")).reshape(len(Rs), -1).T
    return np.array([func(*p) for p in pts]).reshape([2 * R + 1 for R in Rs])
def dft(values, Rs):
    """The coefficients c_l, |l_j| <= R_j, from the values on the grid."""
    c_ = values.astype(complex)
    for j, (R, ax) in enumerate(zip(Rs, grid_axes(Rs))):
        E = np.exp(-1j * np.outer(np.arange(-R, R + 1), ax)) / (2 * R + 1)
        c_ = np.moveaxis(np.tensordot(E, c_, axes=([1], [j])), 0, j)
    return c_
def evaluate(c_, Rs, p):
    e = c_
    for j, R in enumerate(Rs):
        e = np.tensordot(np.exp(1j * np.arange(-R, R + 1) * p[j]), e, axes=([0], [0]))
    return np.real(e)

# a parameter shared by three Pauli rotations that do not commute, with fixed gates between them: degree 3
def rand_u(k):
    z = rgt.normal(size=(k, k)) + 1j * rgt.normal(size=(k, k)); q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))
Xp, Yp, Zp, I2 = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]), np.eye(2)
gens = [np.kron(Xp, Yp) / 2, np.kron(Zp, I2) / 2, np.kron(Yp, Zp) / 2, np.kron(I2, Xp) / 2]
which = [0, 0, 1, 0]                         # the parameter of each gate: theta_1 in three gates, theta_2 in one
fixed = [rand_u(4) for _ in gens]
O4 = rgt.normal(size=(4, 4)) + 1j * rgt.normal(size=(4, 4)); O4 = O4 + O4.conj().T
psi4 = rgt.normal(size=4) + 1j * rgt.normal(size=4); psi4 /= np.linalg.norm(psi4)
def C_shared(t1, t2):
    u = psi4
    for G, V, w_ in zip(gens, fixed, which):
        u = V @ (expm(-1j * (t1, t2)[w_] * G) @ u)
    return np.real(np.vdot(u, O4 @ u))
c_sh = dft(on_grid(C_shared, (3, 1)), (3, 1))
c_sh2 = dft(on_grid(C_shared, (2, 1)), (2, 1))
err_sh = max(abs(evaluate(c_sh, (3, 1), p) - C_shared(*p)) for p in rgt.uniform(-np.pi, np.pi, size=(30, 2)))
err_sh2 = max(abs(evaluate(c_sh2, (2, 1), p) - C_shared(*p)) for p in rgt.uniform(-np.pi, np.pi, size=(30, 2)))
assert err_sh < 1e-12 and err_sh2 > 1e-3
print(f"trigonometric polynomial: a parameter in three non-commuting Pauli rotations; from 7 x 3 grid values "
      f"the error is {err_sh:.1e}, from 5 x 3 (degree 2) {err_sh2:.3f}")

# along the shared parameter, with the other at 0: f has degree 3; 7 values determine it, 5 values alias the
# frequency 3 onto -2; and from 7 estimates with M copies each, the reconstruction stays within sqrt(7) eta
def f_sh(x): return C_shared(x, 0.0)
lamO, vecO = np.linalg.eigh(O4)
def state_sh(x):
    u = psi4
    for G, V, w_ in zip(gens, fixed, which):
        u = V @ (expm(-1j * (x, 0.0)[w_] * G) @ u)
    return u
def interp1(xs_, vs_, R, x):
    l = np.arange(-R, R + 1)
    c_ = np.exp(-1j * np.outer(l, xs_)) @ vs_ / len(xs_)
    return np.real(np.exp(1j * np.outer(np.atleast_1d(x), l)) @ c_)
x7 = 2 * np.pi * np.arange(-3, 4) / 7; x5 = 2 * np.pi * np.arange(-2, 3) / 5
v7 = np.array([f_sh(x) for x in x7]); v5 = np.array([f_sh(x) for x in x5])
cf = np.exp(-1j * np.outer(np.arange(-3, 4), x7)) @ v7 / 7
Msh = 1000
pr7 = [np.abs(vecO.conj().T @ state_sh(x))**2 for x in x7]
rgs = np.random.default_rng(2030)            # its own generator, so that the samples below do not change
est7 = np.array([rgs.multinomial(Msh, p / p.sum()) @ lamO / Msh for p in pr7])
se7 = np.array([np.sqrt((p @ lamO**2 - (p @ lamO)**2) / Msh) for p in pr7])
eta7 = np.max(np.abs(est7 - v7))
xf = np.linspace(-np.pi, np.pi, 361)
ff = np.array([f_sh(x) for x in xf]); f2 = interp1(x5, v5, 2, xf); fh = interp1(x7, est7, 3, xf)
assert np.max(np.abs(interp1(x7, v7, 3, xf) - ff)) < 1e-12 and np.max(np.abs(fh - ff)) <= np.sqrt(7) * eta7
r_true = minimize(lambda p: f_sh(p[0]), [xf[np.argmin(ff)]], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
r_hat = minimize(lambda p: interp1(x7, est7, 3, p[0])[0], [xf[np.argmin(fh)]], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12})
with open(f"{OUT}/trig_shared.dat", "w") as fh_:
    fh_.write("x f f2 fhat lo hi\n")
    for row in zip(xf, ff, f2, fh, ff - np.sqrt(7) * eta7, ff + np.sqrt(7) * eta7):
        fh_.write(" ".join(f"{v:.6f}" for v in row) + "\n")
with open(f"{OUT}/trig_band.dat", "w") as fh_:          # the band f -+ sqrt(7) eta as a closed polygon
    fh_.write("x y\n")
    for xv, yv in list(zip(xf, ff - np.sqrt(7) * eta7)) + list(zip(xf[::-1], (ff + np.sqrt(7) * eta7)[::-1])):
        fh_.write(f"{xv:.6f} {yv:.6f}\n")
with open(f"{OUT}/trig_samples7.dat", "w") as fh_:
    fh_.write("x f est se\n")
    for row in zip(x7, v7, est7, se7):
        fh_.write(" ".join(f"{v:.6f}" for v in row) + "\n")
with open(f"{OUT}/trig_samples5.dat", "w") as fh_:
    fh_.write("x f\n")
    for row in zip(x5, v5):
        fh_.write(" ".join(f"{v:.6f}" for v in row) + "\n")
print(f"  f(x) = C(x, 0): |c_l| for l = 0..3: {np.round(np.abs(cf[3:]), 3)}; range [{ff.min():.3f}, {ff.max():.3f}]; "
      f"width of O {lamO[-1] - lamO[0]:.3f}; the degree-2 interpolant from 5 values is off by up to {np.max(np.abs(f2 - ff)):.3f}")
print(f"  from 7 estimates with M = {Msh}: largest error at the grid eta = {eta7:.3f}, standard errors {se7.min():.3f}..{se7.max():.3f}; "
      f"sup |f^ - f| = {np.max(np.abs(fh - ff)):.3f} <= sqrt(7) eta = {np.sqrt(7) * eta7:.3f}; minimiser {r_true.x[0]:.3f} "
      f"(f = {r_true.fun:.3f}), of f^ {r_hat.x[0]:.3f} (f^ = {r_hat.fun:.3f})")

# the worked example, in y = 2 beta (R = 4) and x = 2 gamma (R = 2): 9 x 5 = 45 values on the grid
RW = (4, 2)
def F_yx(y, x): return F(y / 2, x / 2)
cw = dft(on_grid(F_yx, RW), RW)
for p in rgt.uniform(-np.pi, np.pi, size=(20, 2)):
    assert abs(evaluate(cw, RW, p) - F_yx(*p)) < 1e-12
nz = {(l1 - 4, l2 - 2): np.round(cw[l1, l2], 6) for l1 in range(9) for l2 in range(5) if abs(cw[l1, l2]) > 1e-10}
print(f"  the worked example from its 45 values on the grid; nonzero c_l: {nz}")
for b_, g_ in rgt.uniform(-np.pi, np.pi, size=(5, 2)):     # a certificate of degree (2, 1) that F >= -3
    sos = abs(np.exp(2j * g_) + np.exp(4j * b_))**2 / 4 + abs(1 - np.exp(2j * (2 * b_ + g_)))**2 / 4
    assert abs(F(b_, g_) + 3 - sos) < 1e-12

# the reconstruction from estimates with M copies at each point of the grid, and the lemma on its error
pts_w = np.array(np.meshgrid(*grid_axes(RW), indexing="ij")).reshape(2, -1).T
pgrid = np.array([np.abs(circuit(*angles(y / 2, x / 2)))**2 for y, x in pts_w])
fine = np.linspace(0, 2 * np.pi, 241)[:-1]
Yf, Xf = np.meshgrid(fine, fine, indexing="ij")
F_fine = -2 + np.sin(2 * Yf) * np.sin(Xf)
E2 = [np.exp(1j * np.outer(np.arange(-R, R + 1), fine)) for R in RW]
def on_fine(c_): return np.real(np.einsum("ab,ai,bj->ij", c_, E2[0], E2[1]))
minimisers = np.array([[np.pi / 8, 3 * np.pi / 4], [5 * np.pi / 8, 3 * np.pi / 4], [3 * np.pi / 8, np.pi / 4], [7 * np.pi / 8, np.pi / 4]])
def torus_dist(a, b):
    d_ = np.abs((a - b + np.pi / 2) % np.pi - np.pi / 2)          # (beta, gamma) has period pi in both
    return np.hypot(*d_)
def reconstruct(Mc):
    est = np.array([rgt.multinomial(Mc, p / p.sum()) @ vals / Mc for p in pgrid]).reshape(9, 5)
    c_ = dft(est, RW)
    Ff = on_fine(c_)
    k = np.unravel_index(np.argmin(Ff), Ff.shape)
    r = minimize(lambda p: evaluate(c_, RW, p), [fine[k[0]], fine[k[1]]], method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12})
    bg = np.array([r.x[0] / 2, r.x[1] / 2]) % np.pi
    sup = np.max(np.abs(Ff - F_fine)); maxe = np.max(np.abs(est - on_grid(F_yx, RW)))
    assert sup <= np.sqrt(45) * maxe + 1e-12
    return c_, est, sup, maxe, r.fun, min(torus_dist(bg, m) for m in minimisers), bg
runs = [reconstruct(1000) for _ in range(200)]
sup_, maxe_, fmin_, dist_ = (np.array([r_[i] for r_ in runs]) for i in (2, 3, 4, 5))
print(f"  45 values from M = 1000 copies each (45000 copies): median largest error at the grid {np.median(maxe_):.4f}, "
      f"median sup |F^ - F| {np.median(sup_):.4f} (lemma: at most sqrt(45) times the former); "
      f"median |min F^ + 3| {np.median(np.abs(fmin_ + 3)):.4f}, median distance of its minimiser from one of F {np.median(dist_):.4f}")
c1, est1 = runs[0][0], runs[0][1]
print(f"  first run: min F^ = {runs[0][4]:.4f} at (beta, gamma) = ({runs[0][6][0]:.4f}, {runs[0][6][1]:.4f})")

# the copies that the lemma and Hoeffding's inequality require, for the worked example (w = 4)
eps_, dlt_, G_, w_ = 0.1, 0.05, 45, 4.0
for name, eta in (("lemma", eps_ / (2 * np.sqrt(G_))), ("coefficient by coefficient", eps_ / (2 * G_))):
    Mh = int(np.ceil(w_**2 / (2 * eta**2) * np.log(2 * G_ / dlt_)))
    print(f"  guarantee eps = {eps_}, delta = {dlt_} ({name}): eta = {eta:.5f}, M = {Mh:.3e} copies per point, {G_ * Mh:.3e} in all")

# the sum-of-squares hierarchy: max lambda such that C - lambda = v(z)^* Q v(z), Q >= 0, monomials of degree <= r_j
def sos_bound(c_, Rs, rs):
    mons = [m for m in np.ndindex(*[r + 1 for r in rs])]
    n_ = len(mons)
    Q = cp.Variable((n_, n_), hermitian=True); lam = cp.Variable()
    groups = {}
    for a, ma in enumerate(mons):
        for b, mb in enumerate(mons):
            groups.setdefault(tuple(np.subtract(mb, ma)), []).append((a, b))
    cons = [Q >> 0]
    for l, ab in groups.items():
        inside = all(abs(l_) <= R for l_, R in zip(l, Rs))
        target = c_[tuple(l_ + R for l_, R in zip(l, Rs))] if inside else 0
        if l == (0,) * len(l):
            target = target - lam
        cons.append(sum(Q[a, b] for a, b in ab) == target)
    cp.Problem(cp.Maximize(lam), cons).solve(solver="SCS", eps=1e-9, max_iters=200000)
    return lam.value
for rs in [(2, 1), (3, 1), (2, 2), (4, 2)]:
    print(f"  sum-of-squares bound for F at level {rs}: {sos_bound(cw, RW, rs):.6f}")
for rs in [(4, 2), (5, 3), (6, 4), (8, 4)]:
    print(f"  sum-of-squares bound for F^ of the first run at level {rs}: {sos_bound(c1, RW, rs):.6f} (its minimum {runs[0][4]:.6f})")

# --------------------------------------------------------------------------- SPSA and smoothing
# at the point (0.2, -0.3) of the landscape of the worked example, with c = 0.1, without shot noise
cc = 0.1
th0 = np.array([0.2, -0.3])
def grad_closed(t): return np.array([4 * np.cos(4 * t[0]) * np.sin(2 * t[1]), 2 * np.sin(4 * t[0]) * np.cos(2 * t[1])])
grad = grad_closed(th0)
est = {}
for D in [(1, 1), (1, -1)]:
    D = np.array(D, float)
    dd = (F_closed(*(th0 + cc * D)) - F_closed(*(th0 - cc * D))) / (2 * cc)
    est[tuple(D)] = dd / D
    print(f"SPSA at {th0}, c = {cc}, Delta = {D}: F+ = {F_closed(*(th0 + cc*D)):.4f}, F- = {F_closed(*(th0 - cc*D)):.4f}, "
          f"estimate {np.round(dd / D, 4)}")
mean_est = (est[(1.0, 1.0)] + est[(1.0, -1.0)]) / 2
print(f"  F = {F_closed(*th0):.4f}; mean over Delta {np.round(mean_est, 4)}, gradient {np.round(grad, 4)}, "
      f"difference {np.round(mean_est - grad, 4)}")
print(f"  directional derivatives: along (1,1) {grad @ [1, 1]:.4f}, along (1,-1) {grad @ [1, -1]:.4f}")
print(f"  squared error of one estimate, averaged over Delta: {np.mean([np.sum((e - grad)**2) for e in est.values()]):.4f}"
      f" = |grad|^2 (d - 1) to O(c^2): {np.sum(grad**2):.4f}")
# Gaussian smoothing of the worked example: F_c = -2 + exp(-10 c^2) (F + 2), and the two-point estimator
cs = 0.2
Z = rng.normal(size=(400000, 2))
mc = np.mean(F_closed(th0[0] + cs * Z[:, 0], th0[1] + cs * Z[:, 1]))
print(f"smoothing, c = {cs}: Monte Carlo {mc:.4f}, formula {-2 + np.exp(-10*cs**2) * (F_closed(*th0) + 2):.4f}")
assert abs(mc - (-2 + np.exp(-10 * cs**2) * (F_closed(*th0) + 2))) < 5e-3
Zs = Z[:200000]
vals2 = (F_closed(th0[0] + cs * Zs[:, 0], th0[1] + cs * Zs[:, 1]) - F_closed(*th0)) / cs
gs = np.mean(vals2[:, None] * Zs, axis=0)
print(f"  two-point estimate, mean of 200000: {np.round(gs, 4)}; gradient of F_c: {np.round(np.exp(-10*cs**2)*grad, 4)}")

# --------------------------------------------------------------------------- error against the step
def rmse_fd(h, N):
    bias = (f(phi0 + h) - f(phi0 - h)) / (2 * h) - d_true
    v = (var(phi0 + h) + var(phi0 - h)) / (4 * h**2 * (N / 2))
    return np.sqrt(bias**2 + v)
def rmse_sr(s, N):
    return np.sqrt((var(phi0 + s) + var(phi0 - s)) / (4 * np.sin(s)**2 * (N / 2)))

hs = np.exp(np.linspace(np.log(0.01), np.log(np.pi / 2), 120))
Ns = [2e3, 2e4, 2e5]
with open(f"{OUT}/step_rmse.dat", "w") as fh:
    fh.write("h " + " ".join(f"fd{i} sr{i}" for i in range(len(Ns))) + "\n")
    for h in hs:
        fh.write(f"{h:.6f} " + " ".join(f"{rmse_fd(h, N):.6e} {rmse_sr(h, N):.6e}" for N in Ns) + "\n")

def sampled(h, N, reps=4000, kind="fd"):
    out = []
    pp = np.abs(circuit(np.r_[phi0 + h, tz0[1:]], tx0))**2
    pm = np.abs(circuit(np.r_[phi0 - h, tz0[1:]], tx0))**2
    k = int(N // 2)
    cp = rng.multinomial(k, pp / pp.sum(), size=reps) @ vals / k
    cm = rng.multinomial(k, pm / pm.sum(), size=reps) @ vals / k
    est_ = (cp - cm) / (2 * h if kind == "fd" else 2 * np.sin(h))
    return np.sqrt(np.mean((est_ - d_true)**2))

hsamp = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, np.pi / 2]
with open(f"{OUT}/step_sampled.dat", "w") as fh:
    fh.write("h fd sr\n")
    for h in hsamp:
        fh.write(f"{h:.6f} {sampled(h, 2e3, kind='fd'):.6e} {sampled(h, 2e3, kind='sr'):.6e}\n")

Ncs = np.exp(np.linspace(np.log(1e2), np.log(1e7), 41))
with open(f"{OUT}/copies_rmse.dat", "w") as fh:
    fh.write("N fd hbest sr\n")
    for N in Ncs:
        r = minimize_scalar(lambda lh: rmse_fd(np.exp(lh), N), bounds=(np.log(1e-3), np.log(np.pi / 2)), method="bounded")
        fh.write(f"{N:.6e} {r.fun:.6e} {np.exp(r.x):.6f} {rmse_sr(np.pi / 2, N):.6e}\n")
for N in (2e3, 2e4, 2e5, 2e6):
    r = minimize_scalar(lambda lh: rmse_fd(np.exp(lh), N), bounds=(np.log(1e-3), np.log(np.pi / 2)), method="bounded")
    print(f"N = {N:.0e}: central difference, best step {np.exp(r.x):.3f}, error {r.fun:.4f}; "
          f"shift rule error {rmse_sr(np.pi/2, N):.4f}; central difference at h = 1e-2: {rmse_fd(1e-2, N):.3f}, at 1e-3: {rmse_fd(1e-3, N):.3f}")
print(f"third derivative f''' = -f' = {-d_true:.4f}; the bias of the central difference is f'(sin h / h - 1)")

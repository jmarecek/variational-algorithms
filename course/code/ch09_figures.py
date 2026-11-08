"""Numerical data for the figures and the numbers quoted in Lecture 9 (per-iteration complexity).

The round trip is that of the hardware run of Marecek et al. (2026): depth-four QAOA on 100 qubits on
ibm_boston, in a session, with 20 000 copies a round. The queue is a single first-come-first-served
server shared with other users, whose jobs arrive as a Poisson process; their service times are
illustrative, not measured on any device. The variational examples use the worked example of Lecture 5,
depth-one QAOA for MaxCut on the 4-cycle, in the conventions of code/ch08_figures.py: the cost Hamiltonian
is minus the cut, F(beta, gamma) = -2 + sin 4 beta sin 2 gamma, the parameter-shift rule is applied gate by
gate with 16 circuits, and every method starts at (0.1, -0.1).

Writes, into figures/ch09/:
  queue_timeline.dat  the first hour of one stream of jobs at rho = 0.8 with the mixture of jobs and sessions, served
                    in a session (lane 2), first come first served (lane 1), and with the others first (lane 0): the
                    start and length in minutes, and who: 0 another user, 1 a round of the client, 2 its session
  queue_waits.dat   rho, the mean waits of the Pollaczek-Khinchine formula for the two service distributions, the
                    simulated mean waits of Poisson arrivals, the client's first wait and the mean of its last 20 of 80
                    first come first served, its first and later waits with the others first (mixture), and the
                    bounds rho t_c / (1 - rho) and (rho t_c - t_net) / (1 - rho) on the waits in the long run
  queue_rounds.dat  k and the mean wait of round k at rho = 0.8, first come first served and with the others first,
                    with the bounds of the recursion E W_(k+1) <= rho (E W_k + t_c) and >= rho (E W_k + t_c) - t_net
  copies.dat        M, the step, the iterations measured on the worked example and their bound K(M), and the
                    times K (t_lat + 16 M t_shot) for t_lat = 0 and 6 s, measured and bounded
  rounds.dat        the method, the time and the mean of F - F* over the runs after each round: 0 gradient descent
                    with the shift rule, its 16 circuits submitted together, 1 one by one, 2 simultaneous perturbation,
                    3 COBYLA, 4 the reconstruction from the grid of Lecture 7 (one round)
and prints the numbers quoted in the text.
"""
import os
import numpy as np
from scipy.optimize import minimize

OUT = "figures/ch09"
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------- the round trip
# Marecek et al. (2026), App. "Training QAOA angles on hardware": 20 000 copies a round, in a session; the user's
# wall-clock time 8.2 s a round, the backend's completion time 5.8 s; 100 us a copy; the training took 11.4 minutes
t_user, t_backend, M_hw, t_shot = 8.2, 5.8, 20000, 100e-6
t_net = t_user - t_backend
t_shots = M_hw * t_shot
t_prep = t_backend - t_shots
print(f"the round trip on ibm_boston: {t_user} s a round, of which the backend {t_backend} s: {t_shots:.1f} s of copies "
      f"({M_hw} at {t_shot * 1e6:.0f} us) and {t_prep:.1f} s of preparation; {t_net:.1f} s for the network and the client; "
      f"the device samples {t_shots / t_user:.2f} of the time; 11.4 minutes are {11.4 * 60 / t_user:.0f} rounds; "
      f"{M_hw / t_user:.0f} copies a second, against {1 / t_shot:.0f} without the overheads")

# --------------------------------------------------------------------------- the queue
# the jobs of the other users: Poisson arrivals at the rate lambda = rho / E S, with the service time S either
# exponential with the mean 60 s, or a mixture of jobs (probability 0.9, exponential with the mean 30 s) and of
# sessions (probability 0.1, exponential with the mean 330 s), also with the mean 60 s
rng_q = np.random.default_rng(2040)
ES = 60.0
ES2 = {"exp": 2 * 60.0**2, "mix": 0.9 * 2 * 30.0**2 + 0.1 * 2 * 330.0**2}
def pk(rho, kind):
    """The mean wait of the Pollaczek-Khinchine formula, lambda E S^2 / (2 (1 - rho))."""
    return rho / ES * ES2[kind] / (2 * (1 - rho))

class Stream:
    """The jobs of the other users, from a generator of their own, so that the same stream can be served in
    several ways; every job takes the same draws, whatever the order in which the jobs are served."""
    def __init__(self, rho, kind, seed):
        self.g, self.rho, self.kind = np.random.default_rng(seed), rho, kind
        self.t = self.g.exponential(ES / rho)
    def pop(self):
        g = self.g
        if self.kind == "exp":
            s = g.exponential(60.0)
        else:
            s = g.exponential(330.0) if g.random() < 0.1 else g.exponential(30.0)
        t = self.t; self.t += g.exponential(ES / self.rho)
        return t, s
st_ = Stream(0.5, "mix", 1); sv_ = np.array([st_.pop()[1] for _ in range(10**6)])
assert abs(sv_.mean() - ES) < 0.6 and abs(np.mean(sv_**2) / ES2["mix"] - 1) < 0.03

def mg1_wait(rho, kind, n=10**6, burn=10**5):
    """The mean wait of Poisson arrivals, by Lindley's recursion W' = max(0, W + S - A)."""
    st = Stream(rho, kind, int(rng_q.integers(2**31))); t_prev, w, acc = 0.0, 0.0, 0.0
    t_prev, s_prev = st.pop()
    for i in range(n):
        t, s = st.pop()
        w = max(0.0, w + s_prev - (t - t_prev)); t_prev, s_prev = t, s
        if i >= burn: acc += w
    return acc / (n - burn)

t_up = t_down = t_net / 2              # the network, split evenly between the two directions
s_ours = t_backend                     # the time for which one round occupies the device
t_c = s_ours + t_net                   # a round trip without waiting
def client(rho, kind, mode, N=80, seed=0, burn=10**5, window=None):
    """A client that submits N rounds one after another, each t_net after the previous one has finished, to a device
    that serves the other users' jobs and the rounds either first come, first served ("fcfs"), or the other users
    first ("last"), or in a session that holds the device from the start of the first round to the end of the last
    ("session"). Returns the waits of the rounds, the total time and, if window is given, the jobs that start
    within window seconds of the first submission: (start, length, who), who = 0 others, 1 a round, 2 the session."""
    st = Stream(rho, kind, seed); free = 0.0
    nt, ns = st.pop()
    while nt < burn:                   # bring the queue to its stationary state
        free = max(free, nt) + ns; nt, ns = st.pop()
    log, waits = [], []
    def rec(t_, l_, who):
        if window is not None and t_ < burn + window: log.append(((t_ - burn) / 60, l_ / 60, who))
    t_arr = burn + t_up
    for k in range(N):
        if mode == "last":             # the others that arrive before the device could take the round go first
            while nt <= max(free, t_arr):
                s0 = max(free, nt); free = s0 + ns; rec(s0, ns, 0); nt, ns = st.pop()
        else:
            while nt < t_arr:
                s0 = max(free, nt); free = s0 + ns; rec(s0, ns, 0); nt, ns = st.pop()
        s0 = max(free, t_arr); waits.append(s0 - t_arr)
        if mode == "session":          # the device is held for all the rounds, idle while the results travel
            L_ses = N * s_ours + (N - 1) * t_net
            for j_ in range(N): rec(s0 + j_ * t_c, s_ours, 1)
            rec(s0, L_ses, 2); free = s0 + L_ses
            waits += [0.0] * (N - 1); t_arr = free + t_up
            break
        free = s0 + s_ours; rec(s0, s_ours, 1)
        t_arr = free + t_down + t_up   # the results return, and the next round is submitted at once
    if window is not None:             # the other users' jobs up to the end of the window
        while nt < burn + window:
            s0 = max(free, nt); free = s0 + ns; rec(s0, ns, 0); nt, ns = st.pop()
    return np.array(waits), t_arr - t_up + t_down - burn, log

# the timeline: one stream at rho = 0.8 with the mixture, served in the three ways; the stream is the first, in the
# order of the seeds, in which the first round waits between 0.75 and 1.25 times the mean wait of the formula
seed = 0
while not 0.75 * pk(0.8, "mix") <= client(0.8, "mix", "fcfs", N=1, seed=seed)[0][0] <= 1.25 * pk(0.8, "mix"):
    seed += 1
tl, ends = {}, []
for lane, mode in ((2, "session"), (1, "fcfs"), (0, "last")):
    w_, T_, _ = client(0.8, "mix", mode, seed=seed)
    tl[lane] = (mode, w_, T_); ends.append(T_)
win = 60 * np.ceil(max(ends) / 60 / 30) * 30           # the window, a multiple of half an hour
with open(f"{OUT}/queue_timeline.dat", "w") as fh:
    fh.write("start len lane who\n")
    for lane, (mode, w_, T_) in tl.items():
        for a_, l_, who in client(0.8, "mix", mode, seed=seed, window=win)[2]:
            if a_ + l_ > 0: fh.write(f"{a_:.4f} {l_:.4f} {lane} {who}\n")
        print(f"the timeline at rho = 0.8, mixture, stream {seed}, {mode}: the first wait {w_[0]:.0f} s, the second "
              f"{w_[1]:.0f} s, the mean of the last 40 {w_[40:].mean():.1f} s; the 80 rounds end after {T_ / 60:.1f} min")
print(f"  the window of the timeline: {win / 60:.0f} min")

# the waits round by round at rho = 0.8, and against the load: the first round, the last 20 of 80 first come first
# served, and the rounds after the first with the others first
def round_means(r_, mode, reps, N, bn):
    return np.mean([client(r_, "mix", mode, N=N, seed=int(rng_q.integers(2**31)), burn=bn)[0] for _ in range(reps)], 0)
rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
rows, per_round = [], None
for r_ in rhos:
    bn = 10**5 if r_ < 0.85 else 3 * 10**5
    fc = round_means(r_, "fcfs", 4000 if r_ == 0.8 else (2000 if r_ > 0.8 else 1000), 80, bn)
    la = round_means(r_, "last", 4000 if r_ == 0.8 else 300, 80 if r_ == 0.8 else 20, bn)
    if r_ == 0.8: per_round = (fc, la)
    row = [r_, pk(r_, "exp"), pk(r_, "mix"), mg1_wait(r_, "exp"), mg1_wait(r_, "mix"), fc[0], fc[60:].mean(),
           la[0], la[1:].mean(), r_ * t_c / (1 - r_), max(0.0, (r_ * t_c - t_net) / (1 - r_))]
    rows.append(row)
    print(f"  rho = {r_}: P-K {row[1]:.1f} s and {row[2]:.1f} s, simulated {row[3]:.1f} s and {row[4]:.1f} s "
          f"(exponential, mixture); the client, mixture, first come first served: the first round {row[5]:.1f} s, the "
          f"last 20 {row[6]:.1f} s, between the bounds {row[10]:.1f} and {row[9]:.1f}; the others first: the first round "
          f"{row[7]:.0f} s, the later {row[8]:.1f} s")
with open(f"{OUT}/queue_waits.dat", "w") as fh:
    fh.write("rho pkexp pkmix simexp simmix first late lfirst llate hi lo\n")
    for row in rows:
        fh.write(" ".join(f"{x:.4f}" for x in row) + "\n")
fc, la = per_round
hi, lo = [fc[0]], [fc[0]]
for k in range(79):
    hi.append(0.8 * (hi[-1] + t_c)); lo.append(max(0.0, 0.8 * (lo[-1] + t_c) - t_net))
with open(f"{OUT}/queue_rounds.dat", "w") as fh:
    fh.write("k fcfs last hi lo\n")
    for k in range(80):
        fh.write(f"{k + 1} {fc[k]:.3f} {la[k]:.3f} {hi[k]:.3f} {lo[k]:.3f}\n")
# the means of single rounds fluctuate by about a tenth, those of all the rounds after the first much less
assert 0.95 * np.sum(lo[1:]) <= fc[1:].sum() <= 1.05 * np.sum(hi[1:])
print(f"round by round at rho = 0.8, first come first served: {', '.join(f'{x:.0f}' for x in fc[:6])} s for the first "
      f"six rounds, {fc[60:].mean():.1f} s for the last 20; the bounds {hi[1]:.0f} and {lo[1]:.0f} s for the second "
      f"round, ratio rho = 0.8 a round; with the others first {la[0]:.0f} s, then {la[1:].mean():.1f} s")
r8 = rows[rhos.index(0.8)]
T_fc = fc.sum() + 80 * t_c; T_la = la.sum() + 80 * t_c; T_se = fc[0] + 80 * t_c
print(f"at rho = 0.8 (mixture): E S^2 = {ES2['mix']:.0f} s^2 against {ES2['exp']:.0f} s^2 for the exponential, "
      f"{ES2['mix'] / ES2['exp']:.2f} times the wait; 80 rounds take on average {T_se / 60:.0f} min in a session, "
      f"{T_fc / 60:.0f} min first come first served, and {T_la / 60:.0f} min with the others first, against "
      f"{80 * t_c / 60:.1f} min of round trips; of the first come first served, {fc[0] / 60:.0f} min are the first wait "
      f"and {fc[1:].sum() / 60:.0f} min the later waits")

# the cost of a session to the others: the extra waiting of the other users' jobs of the next four hours, with
# and without a session of 80 rounds that starts with the first round, on the same streams
def others_wait(rho, kind, seed, with_session, N=80, burn=10**5, horizon=4 * 3600):
    st = Stream(rho, kind, seed); free, tot, started = 0.0, 0.0, False
    L_ses = N * s_ours + (N - 1) * t_net
    while True:
        a_, s_ = st.pop()
        if with_session and not started and a_ >= burn + t_up:
            free = max(free, burn + t_up) + L_ses; started = True
        s0 = max(free, a_); free = s0 + s_
        if a_ >= burn + horizon: return tot
        if a_ >= burn: tot += s0 - a_
extra = np.mean([others_wait(0.8, "mix", sd, True) - others_wait(0.8, "mix", sd, False)
                 for sd in rng_q.integers(2**31, size=300)])
L_ses = 80 * s_ours + 79 * t_net
print(f"a session of 80 rounds at rho = 0.8 (mixture) holds the device for {L_ses:.0f} s, of which it samples "
      f"{80 * t_shots:.0f} s; it adds {extra / 3600:.1f} h of waiting in all to the other users' jobs of the next "
      f"four hours, {extra / L_ses:.1f} times its length")

# --------------------------------------------------------------------------- the worked example
n, edges = 4, [(0, 1), (1, 2), (2, 3), (3, 0)]
bits = np.array([[(x >> (n - 1 - q)) & 1 for q in range(n)] for x in range(2**n)])
z = 1 - 2 * bits
zz = np.array([z[:, i] * z[:, j] for i, j in edges])
cuts = np.array([sum(b[i] != b[j] for i, j in edges) for b in bits])
vals = -cuts.astype(float)
cutvals = np.array([0.0, -2.0, -4.0])
onehot = np.array([[c == k for k in (0, 2, 4)] for c in cuts], float)

def probs(tz, tx):
    """Probabilities of the 16 bit strings; tz, tx of shape (R, 4): the angles of the R_Z gates of the edges and of
    the R_X gates of the mixer."""
    R = tz.shape[0]
    psi = (np.exp(-0.5j * tz @ zz) / 4).reshape((R,) + (2,) * n)
    for q in range(n):
        c, s = np.cos(tx[:, q] / 2), -1j * np.sin(tx[:, q] / 2)
        U = np.stack([np.stack([c, s], -1), np.stack([s, c], -1)], -2)
        m = np.moveaxis(psi, q + 1, -1)
        psi = np.moveaxis(np.einsum("rab,r...b->r...a", U, m), -1, q + 1)
    return (np.abs(psi)**2).reshape(R, -1)
def F_closed(beta, gamma): return -2 + np.sin(4 * beta) * np.sin(2 * gamma)
def grad_closed(beta, gamma):
    return np.array([4 * np.cos(4 * beta) * np.sin(2 * gamma), 2 * np.sin(4 * beta) * np.cos(2 * gamma)])
def angles(beta, gamma):
    beta, gamma = np.atleast_1d(beta), np.atleast_1d(gamma)
    return np.repeat(gamma[:, None], 4, 1), np.repeat(2 * beta[:, None], 4, 1)
for b_, g_ in [(0.1, -0.1), (0.7, -0.4), (0.2, -0.3)]:
    assert abs(probs(*angles(b_, g_)) @ vals - F_closed(b_, g_)) < 1e-12

def sample_mean(pc, M, g):
    """Means of M measured values with the probabilities pc (R, 3) of the cuts 0, 2, 4."""
    pc = np.clip(pc, 0, 1)
    n0 = g.binomial(M, pc[:, 0])
    r1 = np.where(pc[:, 0] < 1, pc[:, 1] / np.maximum(1 - pc[:, 0], 1e-300), 0)
    n1 = g.binomial(M - n0, np.clip(r1, 0, 1))
    return (cutvals[1] * n1 + cutvals[2] * (M - n0 - n1)) / M
def F_est(beta, gamma, M, g):
    return sample_mean(probs(*angles(beta, gamma)) @ onehot, M, g)
S = np.pi / 2
def psr(beta, gamma, M, g):
    """The gradient estimated gate by gate, M copies of each of the 16 circuits, for all the runs in one batch."""
    tz, tx = angles(beta, gamma)
    TZ, TX = np.repeat(tz[None], 16, 0), np.repeat(tx[None], 16, 0)
    for e in range(4):
        TZ[2 * e, :, e] += S; TZ[2 * e + 1, :, e] -= S
        TX[8 + 2 * e, :, e] += S; TX[9 + 2 * e, :, e] -= S
    pc = probs(TZ.reshape(-1, 4), TX.reshape(-1, 4)) @ onehot
    m = sample_mean(pc, M, g).reshape(16, len(beta))
    return np.stack([(m[8::2] - m[9::2]).sum(0), (m[0:8:2] - m[1:8:2]).sum(0) / 2], 1)
pc_ = probs(*angles(np.linspace(0, np.pi, 41), np.linspace(-np.pi, np.pi, 41))) @ onehot
var1 = pc_ @ cutvals**2 - (pc_ @ cutvals)**2
assert var1.max() <= 2 + 1e-9

beta0, gamma0 = 0.1, -0.1
F0, Fstar, Lb = F_closed(beta0, gamma0), -3.0, 20.0
t0 = np.array([beta0, gamma0])

# --------------------------------------------------------------------------- copies per round
# stochastic gradient descent with the shift rule on the worked example: V = V1 / M with V1 = 10 sigma^2 = 20 (Lecture 8),
# epsilon = 0.5, and the step alpha(M) = min(1/L, epsilon M / (4 L V1)), that of the corollary on unbiased estimates for
# K = K(M) = max(8 L D / epsilon, 32 L D V1 / (M epsilon^2)); the iterations measured until the running average over the
# iterations of the mean over the runs of |grad F|^2 is at most epsilon; 16 circuits a round
rng_c = np.random.default_rng(2041)
V1, eps, Dl, ncirc = 20.0, 0.5, F0 - Fstar, 16
Mstar = 4 * V1 / eps
def Kbound(M): return max(8 * Lb * Dl / eps, 32 * Lb * Dl * V1 / (M * eps**2))
Ms = [1, 2, 4, 8, 16, 32, 64, 160, 400, 1000, 4000, 16000]
runs_c = 100
crows = []
for M in Ms:
    al = min(1 / Lb, eps * M / (4 * Lb * V1))
    th = np.repeat(t0[None], runs_c, 0); csum, k = 0.0, 0
    while True:
        csum += np.mean(np.sum(grad_closed(th[:, 0], th[:, 1])**2, axis=0)); k += 1
        if csum / k <= eps or k > 3 * 10**5: break
        th = th - al * psr(th[:, 0], th[:, 1], M, rng_c)
    Kb = Kbound(M)
    crows.append((M, al, k, Kb, k * ncirc * M * t_shot, k * (6 + ncirc * M * t_shot), Kb * ncirc * M * t_shot,
                  Kb * (6 + ncirc * M * t_shot)))
    print(f"  copies M = {M}: step {al:.5f}, {k} iterations measured, bound {Kb:.0f}; time {crows[-1][4]:.1f} s without "
          f"latency, {crows[-1][5]:.0f} s with 6 s a round; the bound's {crows[-1][6]:.1f} s and {crows[-1][7]:.0f} s")
with open(f"{OUT}/copies.dat", "w") as fh:
    fh.write("M step K Kb T0 T6 Tb0 Tb6\n")
    for r_ in crows:
        fh.write(f"{r_[0]} {r_[1]:.6f} {r_[2]} {r_[3]:.1f} {r_[4]:.4f} {r_[5]:.3f} {r_[6]:.4f} {r_[7]:.3f}\n")
i_s = Ms.index(160)
print(f"copies per round: M* = 4 V1 / epsilon = {Mstar:.0f}; K(M*) = {Kbound(Mstar):.0f}; with 6 s a round the bound's "
      f"time is {crows[0][7] / crows[i_s][7]:.0f} times as long at M = 1 as at M*, the measured {crows[0][5] / crows[i_s][5]:.0f} "
      f"times; the formula M* (t + c t_shot) / (t + c M* t_shot) = "
      f"{Mstar * (6 + ncirc * t_shot) / (6 + ncirc * Mstar * t_shot):.0f}; without latency the measured times at "
      f"M = 1 and M* are {crows[0][4]:.1f} s and {crows[i_s][4]:.1f} s")

# --------------------------------------------------------------------------- the methods by rounds
# M = 1000 copies of each circuit, t_lat = 6 s for each submission, 100 us a copy; the mean of F - F* over 50 runs
rng_r = np.random.default_rng(2042)
Mr, t_lat, runs_r = 1000, 6.0, 50
def t_round(c, batched=True): return (t_lat + c * Mr * t_shot) if batched else c * (t_lat + Mr * t_shot)
curves = {}
# gradient descent with the shift rule, the step 1/20, 16 circuits a round
th = np.repeat(t0[None], runs_r, 0); gd = [np.mean(F_closed(th[:, 0], th[:, 1])) - Fstar]
for k in range(60):
    th = th - psr(th[:, 0], th[:, 1], Mr, rng_r) / Lb; gd.append(np.mean(F_closed(th[:, 0], th[:, 1])) - Fstar)
curves["gd"] = np.array(gd)
# simultaneous perturbation, c = 0.1 and the step 1/40, 2 circuits a round
th = np.repeat(t0[None], runs_r, 0); sp = [curves["gd"][0]]
for k in range(400):
    D = rng_r.choice([-1.0, 1.0], size=th.shape)
    y = F_est(*np.vstack([th + 0.1 * D, th - 0.1 * D]).T, Mr, rng_r).reshape(2, -1)
    th = th - ((y[0] - y[1]) / 0.2)[:, None] * D / 40; sp.append(np.mean(F_closed(th[:, 0], th[:, 1])) - Fstar)
curves["spsa"] = np.array(sp)
# COBYLA, one circuit a round: the exact value at the point of the least estimate so far
cob = np.zeros((runs_r, 151))
for r_ in range(runs_r):
    best, bx, hist = np.inf, t0.copy(), []
    def fun(x):
        global best, bx
        v = F_est(np.array([x[0]]), np.array([x[1]]), Mr, rng_r)[0]
        if v < best: best, bx = v, x.copy()
        hist.append(F_closed(*bx) - Fstar)
        return v
    minimize(fun, t0, method="COBYLA", options={"rhobeg": 0.2, "maxiter": 150, "tol": 1e-4})
    h_ = np.array([F0 - Fstar] + hist)[:151]
    cob[r_, :len(h_)] = h_; cob[r_, len(h_):] = h_[-1]
curves["cobyla"] = cob.mean(0)
# the reconstruction of Lecture 7: the 45 circuits of the grid in one round, then the least value of the reconstruction
# on a grid of 721 x 361 points; y = 2 beta has degree at most 4, x = 2 gamma at most 2
ky, kx = np.arange(9), np.arange(5)
Yg, Xg = np.meshgrid(2 * np.pi * ky / 9, 2 * np.pi * kx / 5, indexing="ij")
bf, gf = np.meshgrid(np.linspace(0, np.pi, 721), np.linspace(-np.pi / 2, np.pi / 2, 361), indexing="ij")
lys, lxs = np.arange(-4, 5), np.arange(-2, 3)
Ey = np.exp(1j * np.outer(2 * bf[:, 0], lys)); Ex = np.exp(1j * np.outer(2 * gf[0], lxs))
def reconstruct(values):
    c = np.zeros((9, 5), complex)
    for a, ly in enumerate(lys):
        for b, lx in enumerate(lxs):
            c[a, b] = np.mean(values * np.exp(-1j * (ly * Yg + lx * Xg)))
    return np.real(Ey @ c @ Ex.T)
exact_vals = F_closed(Yg / 2, Xg / 2)
assert np.allclose(reconstruct(exact_vals), F_closed(bf, gf), atol=1e-9)
grid_res = []
for r_ in range(runs_r):
    est = F_est(Yg.ravel() / 2, Xg.ravel() / 2, Mr, rng_r).reshape(9, 5)
    Rc = reconstruct(est); i, j = np.unravel_index(np.argmin(Rc), Rc.shape)
    grid_res.append(F_closed(bf[i, j], gf[i, j]) - Fstar)
grid_res = np.array(grid_res)
tg = t_round(45)
with open(f"{OUT}/rounds.dat", "w") as fh:
    fh.write("m t v\n")
    for m_, (name, c_, bt) in enumerate((("gd", 16, True), ("gdun", 16, False), ("spsa", 2, True), ("cobyla", 1, True))):
        v_ = curves["gd" if name == "gdun" else name]
        for k in range(len(v_)):
            fh.write(f"{m_} {max(k * t_round(c_, bt), 1.0):.3f} {v_[k]:.6e}\n")
    fh.write(f"4 1.000 {F0 - Fstar:.6e}\n4 {tg:.3f} {grid_res.mean():.6e}\n4 6000.000 {grid_res.mean():.6e}\n")
def first_below(v, tr, lev):
    k = np.argmax(v <= lev)
    return (k, k * tr) if v[k] <= lev else (None, None)
print(f"the methods by rounds, M = {Mr}, {t_lat:.0f} s a submission: a round takes {t_round(16):.1f} s for the shift rule "
      f"({t_round(16, False):.1f} s with its circuits one by one), {t_round(2):.1f} s for simultaneous perturbation, "
      f"{t_round(1):.1f} s for COBYLA and {tg:.1f} s for the grid of 45 circuits")
for name, c_, bt in (("gd", 16, True), ("gdun", 16, False), ("spsa", 2, True), ("cobyla", 1, True)):
    v_ = curves["gd" if name == "gdun" else name]
    k1, t1_ = first_below(v_, t_round(c_, bt), 0.01)
    print(f"  {name}: mean F - F* below 0.01 after {k1} rounds, {t1_ if t1_ is None else round(t1_)} s; "
          f"after 10 minutes {v_[min(int(600 // t_round(c_, bt)), len(v_) - 1)]:.4f}; at the end {v_[-1]:.5f}")
print(f"  grid: one round of {tg:.1f} s; F - F* at the minimiser of the reconstruction {grid_res.mean():.5f} on average, "
      f"at most {grid_res.max():.5f} over {runs_r} runs")

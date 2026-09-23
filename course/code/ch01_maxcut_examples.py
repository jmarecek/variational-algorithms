"""MaxCut three ways, for Lecture 1 of "Variational Methods".

1. Exact solution with Gurobi (gurobipy).
2. The Goemans-Williamson SDP relaxation with cvxpy, and random-hyperplane rounding.
3. QAOA with Qiskit: cold start from |+>^n, and a warm start from a GW cut.

Run from the course directory:  python3 code/ch01_maxcut_examples.py
Writes figures to figures/code/ and prints the numbers quoted in the lecture.
"""
import numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n = 12
def make_graph(seed):
    """A random 4-regular graph on n vertices with integer weights in 1..9."""
    r = np.random.default_rng(seed)
    G = nx.random_regular_graph(4, n, seed=seed)
    for (i, j) in G.edges():
        G[i][j]["w"] = int(r.integers(1, 10))
    return G
SEED = 11                                             # chosen below so that the SDP relaxation is not exact
rng = np.random.default_rng(SEED)
G = make_graph(SEED)
W = nx.to_numpy_array(G, weight="w")
edges = [(i, j, G[i][j]["w"]) for i, j in G.edges()]

def cut_value(z):                                    # z in {-1,+1}^n
    return 0.5 * sum(w * (1 - z[i] * z[j]) for i, j, w in edges)

# ---------------------------------------------------------------- 1. Gurobi
import gurobipy as gp
m = gp.Model("maxcut"); m.Params.OutputFlag = 0
x = m.addVars(n, vtype=gp.GRB.BINARY)
# edge (i,j) is cut iff x_i != x_j, i.e. x_i + x_j - 2 x_i x_j = 1
m.setObjective(gp.quicksum(w * (x[i] + x[j] - 2 * x[i] * x[j]) for i, j, w in edges), gp.GRB.MAXIMIZE)
m.optimize()
z_opt = np.array([1 - 2 * round(x[i].X) for i in range(n)])
OPT = m.ObjVal
print(f"Gurobi: OPT = {OPT:.1f}, cut = {z_opt}")

# ---------------------------------------------------------------- 2. cvxpy SDP + GW rounding
import cvxpy as cp
Y = cp.Variable((n, n), symmetric=True)
obj = 0.5 * sum(w * (1 - Y[i, j]) for i, j, w in edges)
sdp = cp.Problem(cp.Maximize(obj), [Y >> 0, cp.diag(Y) == 1])
sdp.solve(solver=cp.SCS, eps=1e-8)
SDP = sdp.value
Yv = Y.value
evals, evecs = np.linalg.eigh(Yv)                    # Y = V^T V, vectors v_i = columns of V
V = (evecs * np.sqrt(np.clip(evals, 0, None))).T
def gw_round(V, rng, trials):
    cuts = []
    for _ in range(trials):
        r = rng.normal(size=V.shape[0])
        z = np.sign(r @ V); z[z == 0] = 1
        cuts.append(cut_value(z))
    return np.array(cuts)
gw_cuts = gw_round(V, rng, 1000)
print(f"cvxpy/SCS: SDP = {SDP:.3f} >= OPT = {OPT:.1f};  OPT/SDP = {OPT/SDP:.4f}")
print(f"GW rounding, 1000 hyperplanes: mean {gw_cuts.mean():.2f} = {gw_cuts.mean()/OPT:.3f} OPT, "
      f"best {gw_cuts.max():.0f}, best-of-10 finds OPT in {np.mean([gw_round(V, rng, 10).max() >= OPT-1e-9 for _ in range(100)]):.0%} of runs")

# ---------------------------------------------------------------- 3. QAOA with Qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
basis = np.array([[1 - 2 * ((k >> q) & 1) for q in range(n)] for k in range(2 ** n)])  # Qiskit: qubit q is bit q of the index
cut_of_basis = np.array([cut_value(z) for z in basis])

def qaoa_circuit(betas, gammas, warm=None, eps=0.25):
    """Depth-p QAOA. warm=None: cold start |+>^n with X mixer.
    warm=z (a cut in {-1,+1}^n): product state with P(x_i=1)=c_i, c_i in {eps,1-eps},
    and the rotated mixer of Egger, Marecek, Woerner (2021) that reproduces z at beta=pi/2, gamma=0."""
    qc = QuantumCircuit(n)
    if warm is None:
        qc.h(range(n)); thetas = None
    else:
        c = np.where(warm < 0, 1 - eps, eps)             # z_i=-1 <-> x_i=1
        thetas = 2 * np.arcsin(np.sqrt(c))
        for q in range(n): qc.ry(thetas[q], q)
    for beta, gamma in zip(betas, gammas):
        for i, j, w in edges: qc.rzz(gamma * w, i, j)     # exp(-i gamma H_C) up to a global phase
        for q in range(n):
            if warm is None: qc.rx(2 * beta, q)
            else: qc.ry(thetas[q], q); qc.rz(-2 * beta, q); qc.ry(-thetas[q], q)   # matrix R_Y(-t) R_Z(-2b) R_Y(t)
    return qc
def probs(betas, gammas, warm=None):
    sv = Statevector(qaoa_circuit(betas, gammas, warm)).data
    p = np.abs(sv) ** 2
    return p
def expected_cut(theta, p, warm=None):
    pr = probs(theta[:p], theta[p:], warm)
    return float(pr @ cut_of_basis)

# depth-one landscape of the cold start (also used to initialise the optimizer)
B = np.linspace(0, np.pi, 41); Gm = np.linspace(0, np.pi, 41)
land = np.array([[expected_cut(np.array([b, g]), 1) for g in Gm] for b in B])

def optimize(p, warm, seeds):
    """COBYLA from several initial points; returns the best (value, probabilities, angles)."""
    best = None
    for th0 in seeds:
        res = minimize(lambda t: -expected_cut(t, p, warm), th0, method="COBYLA", options={"maxiter": 600, "rhobeg": 0.3})
        if best is None or res.fun < best.fun: best = res
    return -best.fun, probs(best.x[:p], best.x[p:], warm), best.x

z_gw = None
for _ in range(50):                                      # one GW cut to warm-start from (a typical, not the best)
    r = rng.normal(size=n); z = np.sign(r @ V); z[z == 0] = 1
    if cut_value(z) < OPT: z_gw = z; break
if z_gw is None:                                         # every GW cut was optimal: start from a random cut instead
    z_gw = rng.choice([-1, 1], size=n)
print(f"warm start from a GW cut of value {cut_value(z_gw):.0f} = {cut_value(z_gw)/OPT:.3f} OPT")

results, prev = {}, {}
ib, ig = np.unravel_index(land.argmax(), land.shape)
for kind, warm in (("cold", None), ("warm", z_gw)):
    for p in (1, 2, 3):
        seeds = [rng.uniform(0, np.pi, 2 * p) for _ in range(6)]
        seeds.append(np.concatenate([np.linspace(0.8, 0.1, p), np.linspace(0.05, 0.4, p)]))     # linear ramp (TQA-like)
        if p == 1:
            seeds.append(np.array([B[ib], Gm[ig]]) if kind == "cold" else np.array([np.pi / 2, 0.0]))
        else:                                              # depth p-1 optimum plus a trivial layer (beta=0 acts as identity)
            bp, gp_ = prev[kind][:p - 1], prev[kind][p - 1:]
            seeds.append(np.concatenate([bp, [0.0], gp_, [0.0]]))
        val, pr, ang = optimize(p, warm, seeds)
        results[(kind, p)] = (val, pr); prev[kind] = ang
        print(f"QAOA {kind} start p={p}: <cut> = {val:.2f} = {val/OPT:.3f} OPT, P[optimum] = {pr[cut_of_basis>=OPT-1e-9].sum():.3f}, "
              f"P[cut >= GW start] = {pr[cut_of_basis>=cut_value(z_gw)-1e-9].sum():.3f}")

# ---------------------------------------------------------------- figures
blue, orange, aqua, gray = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.6))
pos = nx.spring_layout(G, seed=3)
cols = [blue if z_opt[i] > 0 else orange for i in range(n)]
cut_edges = [(i, j) for i, j, w in edges if z_opt[i] != z_opt[j]]
nx.draw_networkx_nodes(G, pos, node_color=cols, node_size=260, ax=ax[0])
nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax[0])
nx.draw_networkx_edges(G, pos, edgelist=[e for e in G.edges() if e not in cut_edges and e[::-1] not in cut_edges], width=1, edge_color=gray, ax=ax[0])
nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2, edge_color="black", style="dashed", ax=ax[0])
nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): w for i, j, w in edges}, font_size=7, ax=ax[0])
ax[0].set_title(f"Gurobi: maximum cut = {OPT:.0f} (dashed edges)", fontsize=10); ax[0].axis("off")
im = ax[1].imshow(Yv, cmap="RdBu", vmin=-1, vmax=1)
ax[1].set_title(f"cvxpy/SCS: SDP optimum Y, value {SDP:.2f}", fontsize=10)
ax[1].set_xlabel("vertex j"); ax[1].set_ylabel("vertex i")
plt.colorbar(im, ax=ax[1], fraction=0.046, label=r"$Y_{ij} = v_i^\top v_j$")
fig.tight_layout(); fig.savefig("figures/code/maxcut_gurobi_sdp.pdf")

fig, ax = plt.subplots(figsize=(8.6, 3.2))
vals, counts = np.unique(gw_cuts, return_counts=True)
ax.bar(vals, counts / counts.sum(), width=0.8, color=blue, label="GW cuts, 1000 hyperplanes")
ax.axvline(OPT, color="black", lw=1.5, label=f"OPT = {OPT:.0f} (Gurobi)")
ax.axvline(SDP, color=orange, lw=1.5, ls="--", label=f"SDP = {SDP:.2f} (cvxpy)")
ax.axvline(0.878 * SDP, color=gray, lw=1.2, ls=":", label=r"$\alpha_{\rm GW}\cdot$SDP" + f" = {0.878*SDP:.2f}")
ax.axvline(gw_cuts.mean(), color=aqua, lw=1.5, label=f"mean GW cut = {gw_cuts.mean():.2f}")
ax.set_xlabel("cut value"); ax.set_ylabel("frequency"); ax.legend(frameon=False, fontsize=8, loc="upper left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("figures/code/maxcut_gw_hist.pdf")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.4))
im = ax[0].imshow(land, origin="lower", extent=[Gm[0], Gm[-1], B[0], B[-1]], aspect="auto", cmap="viridis")
ax[0].set_xlabel(r"$\gamma$"); ax[0].set_ylabel(r"$\beta$"); ax[0].set_title(r"cold-start QAOA, $p=1$: $\langle$cut$\rangle(\beta,\gamma)$", fontsize=10)
plt.colorbar(im, ax=ax[0], fraction=0.046)
edges_hist = np.arange(cut_of_basis.min() - 0.5, cut_of_basis.max() + 1.5, 1.0)
uni = np.histogram(cut_of_basis, bins=edges_hist)[0] / 2 ** n
ax[1].step(edges_hist[:-1] + 0.5, np.cumsum(uni), where="post", color=gray, ls=":", label="uniform")
for (kind, p), col, ls in [(("cold", 1), blue, "-"), (("cold", 3), blue, "--"), (("warm", 1), orange, "-"), (("warm", 3), orange, "--")]:
    pr = results[(kind, p)][1]
    h = np.histogram(cut_of_basis, bins=edges_hist, weights=pr)[0]
    ax[1].step(edges_hist[:-1] + 0.5, np.cumsum(h), where="post", color=col, ls=ls, lw=1.8, label=f"{kind} start, p={p}")
ax[1].axvline(OPT, color="black", lw=1); ax[1].axvline(cut_value(z_gw), color=aqua, lw=1, label="GW cut used as warm start")
ax[1].set_xlabel("cut value of sampled bitstring"); ax[1].set_ylabel("cumulative probability"); ax[1].legend(frameon=False, fontsize=7, loc="upper left")
ax[1].set_title("Qiskit statevector: sampling distribution", fontsize=10)
for s in ["top", "right"]: ax[1].spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("figures/code/maxcut_qaoa.pdf")
print("figures written to figures/code/")

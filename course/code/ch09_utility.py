"""Data for the figures of Lecture 9 on setting the angles of QAOA at utility scale, after Marecek et al. (2026).

The numbers come from two sources. The raw data of the paper are read from a clone of
github.com/Quantum-Working-Groups/QAOA-Parameter-Setting, at the path in the environment variable QAOA_PARAMS_REPO
(by default ../../QAOA-Parameter-Setting, relative to the course): the training of MaxCut and MIS on ten 40-node
3-regular graphs, the energies of Fig. 7 of the paper, and, on its branch adding_landscape_results, the landscapes of
Fig. 8. The figures of the paper for which no data are public (Figs. 5, 9, 10 and 12) are vector PDFs in
figures/numerics/, and their bars, markers and lines are decoded here from the drawing commands, against the tick
labels of their axes; the decoded counts of Fig. 5 are integers out of 37 graphs to within 0.02.

Writes, into figures/ch09/:
  util_best.dat     the depth p and the number of the 37 graphs on which each method gives the best angles, with the
                    statevector (sv_*) and with Pauli propagation (pp_*): fixed angles, Fourier, Interp., linear ramp,
                    recursive transition states, TQA, all with every angle optimised
  util_land.dat     the slopes (dg, db) of depth-five linear-ramp QAOA on a 144-node heavy-hex graph, and the energies
                    measured on ibm_fez (qpu) and computed with MPS (mps) and Pauli propagation (pp)
  util_land_top.dat the boundaries of the top tenth of each landscape, as segments: k = 0 the device, 1 MPS, 2 PP
  util_evals.dat    the depth p and, for TQA with every angle optimised on a 40-node 3-regular graph, the energy that
                    Pauli propagation and MPS report at their optimum (pp_a, mps_a) and the exact energy of the same
                    angles (pp_x, mps_x)
  util_hw.dat       for each method (m, the index in METHODS) and evaluator (ev: 0 MPS, 1 Pauli propagation), the mean
                    estimated and hardware approximation ratios over ten 144-node heavy-hex graphs at the depths 5 and 10
                    (Fig. 9 of the paper), and their changes from 5 to 10
  util_cost.dat     for each method with Pauli propagation, the mean total duration and hardware approximation ratio at
                    the depth 10 (Fig. 10 of the paper); m indexes fixed angles without optimisation, parameter transfer,
                    TQA and linear ramp with their parameters optimised, and fixed angles, linear ramp, TQA, Interp.,
                    Fourier and recursive TS with every angle optimised
  util_pareto.dat   the Pareto frontiers of the emulated approximation ratio against the resource without latency (k = 0)
                    and with 6 s a circuit (k = 1), as segments with the family f: 0 fixed angles, 1 parameter transfer
  util_pareto_pts.dat  the prescriptions marked on the frontiers: k, the resource, the ratio and the method
                    (0 fixed angles with optimised angles, 1 without, 2 parameter transfer)
and prints the numbers quoted in the text.
"""
import io, json, os, glob, subprocess
import numpy as np
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures/ch09"
REPO = os.environ.get("QAOA_PARAMS_REPO", "../../QAOA-Parameter-Setting")
PDF = "figures/numerics"
os.makedirs(OUT, exist_ok=True)

TOL = {(0.267, 0.467, 0.667): "FA", (0.933, 0.4, 0.467): "F", (0.133, 0.533, 0.2): "I", (0.8, 0.733, 0.267): "LR",
       (0.4, 0.8, 0.933): "RTS", (0.667, 0.2, 0.467): "TQA", (0.781, 0.781, 0.781): "PT"}
METHODS = ["FA", "F", "I", "LR", "RTS", "TQA", "PT"]
def rgb(c): return tuple(round(x, 3) for x in c) if c else None
def ticks(page, labels, axis, region):
    """The centres of the tick labels in a region (x0, y0, x1, y1) of the page, as pairs (value, coordinate)."""
    out = []
    for w in page.get_text("words"):
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if region[0] <= cx <= region[2] and region[1] <= cy <= region[3] and w[4] in labels:
            out.append((labels[w[4]], cx if axis == "x" else cy))
    return sorted(out)
def affine(pairs, log=False):
    """The map from page coordinates to data, fitted to the tick labels."""
    v = np.array([np.log10(a) if log else a for a, _ in pairs]); c = np.array([b for _, b in pairs])
    k, m = np.polyfit(c, v, 1)
    res = np.abs(k * c + m - v).max()
    assert res < 0.02 * (v.max() - v.min()), res
    return (lambda z: 10 ** (k * z + m)) if log else (lambda z: k * z + m)

# --------------------------------------------------------------------------- Fig. 5: the best methods
doc = fitz.open(f"{PDF}/figure5.pdf"); pg = doc[0]
yl = ticks(pg, {"0.0": 0.0, "1.0": 1.0}, "y", (0, 0, 40, 115))
H = yl[0][1] - yl[1][1]
best = {}
for d in pg.get_drawings():
    if d.get("fill") is None or len(d["items"]) != 1 or d["items"][0][0] != "re": continue
    m = TOL.get(rgb(d["fill"]))
    if m is None: continue
    r = d["items"][0][1]; cx = (r.x0 + r.x1) / 2
    right, bottom = cx > 132, r.y1 > 115
    if r.y1 > 209 or r.y1 < 10: continue
    p = (cx - (142.2 if right else 42.8)) / 10.04 + 2
    if abs(p - round(p)) > 0.15: continue
    key = ({(0, 0): "sv", (1, 0): "pp", (0, 1): "mpsq", (1, 1): "mpsa"}[(int(right), int(bottom))], int(round(p)))
    best.setdefault(key, {}).setdefault(m, 0.0)
    best[key][m] += r.height / H * 37
for key, row in best.items():
    assert abs(sum(row.values()) - 37) < 0.05 and all(abs(v - round(v)) < 0.02 for v in row.values()), (key, row)
    for m in row: row[m] = int(round(row[m]))
with open(f"{OUT}/util_best.dat", "w") as fh:
    cols = [f"{e}_{m}" for e in ("sv", "pp") for m in METHODS[:6]]
    fh.write("p " + " ".join(cols) + "\n")
    for p in range(2, 11):
        fh.write(f"{p} " + " ".join(str(best[(e, p)].get(m, 0)) for e in ("sv", "pp") for m in METHODS[:6]) + "\n")
for e in ("sv", "pp", "mpsq", "mpsa"):
    print(f"Fig. 5, {e}: " + "; ".join(f"p={p}: " + ", ".join(f"{m} {best[(e, p)][m]}" for m in METHODS[:6]
                                                               if m in best[(e, p)]) for p in range(2, 11)))
for e in ("sv", "pp"):
    tot = {m: sum(best[(e, p)].get(m, 0) for p in range(2, 11)) for m in METHODS[:6]}
    print(f"  {e}, over the 9 depths x 37 graphs = 333: " + ", ".join(f"{m} {tot[m]}" for m in METHODS[:6]))

# --------------------------------------------------------------------------- Fig. 6: MaxCut against MIS
# the energies of ten 40-node 3-regular graphs with Pauli propagation, MaxCut with the largest weight 4 and MIS with 5 (the
# settings whose energies are those of the figure); Fourier and Interp. train depth by depth in one file, for MIS to
# the depth 8
RR = f"{REPO}/data/training/random_regular"
def energy(prob, meth, mw, g, p):
    if meth in ("F", "I"):
        fs = glob.glob(f"{RR}/*_{g:03d}N40R3R_{prob}_{meth}_PP_optMW{mw}_*.json")     # to the depth 10 or 8
        assert len(fs) == 1, fs
        d = json.load(open(fs[0])); last = str(max(int(k) for k in d if k.isdigit()))
        return d[last][str(p)]["energy"]
    fs = glob.glob(f"{RR}/*_{g:03d}N40R3R_{prob}_{meth}_PP_optMW{mw}_{p}.json")
    assert len(fs) == 1, fs
    d = json.load(open(fs[0])); last = str(max(int(k) for k in d if k.isdigit()))
    return d[last]["energy"]
flip = {}
for prob, mw in (("MC", 4), ("MIS", 5)):
    for meth in ("F", "I", "FA", "TQA"):
        for p in (2, 4, 6, 8):
            e = np.array([energy(prob, meth, mw, g, p) for g in range(10)])
            flip[(prob, meth, p)] = (e.mean(), e.std())
names = {"F": "Fourier", "I": "Interp.", "FA": "fixed angles", "TQA": "TQA"}
for prob in ("MC", "MIS"):
    for p in (2, 4, 6, 8):
        row = sorted(((flip[(prob, m, p)][0], m) for m in ("F", "I", "FA", "TQA")), reverse=True)
        top = row[0][0]
        print(f"Fig. 6, {prob} p={p}: " + ", ".join(f"{m} {v:.3f} +- {flip[(prob, m, p)][1]:.3f} "
                                                   f"({100 * (v - top) / abs(top):+.1f}%)" for v, m in row))

# --------------------------------------------------------------------------- Fig. 7: the evaluators against the truth
LC = f"{REPO}/figures/large_case_hpc"
ev = {k: json.load(open(f"{LC}/{k}.json")) for k in ("FA_PP", "FA_MPSAer", "TQA_PP", "TQA_MPSAer")}
with open(f"{OUT}/util_evals.dat", "w") as fh:
    fh.write("p pp_a pp_x mps_a mps_x\n")
    for p in ("2", "4", "6", "8", "10"):
        fh.write(f"{p} {ev['TQA_PP']['approx'][p]:.4f} {ev['TQA_PP']['exact'][p]:.4f} "
                 f"{ev['TQA_MPSAer']['approx'][p]:.4f} {ev['TQA_MPSAer']['exact'][p]:.4f}\n")
for m in ("TQA", "FA"):
    print(f"Fig. 7, {m}: " + "; ".join(f"p={p}: PP {ev[m + '_PP']['approx'][p]:.2f} -> {ev[m + '_PP']['exact'][p]:.2f}, "
                                       f"MPS {ev[m + '_MPSAer']['approx'][p]:.2f} -> {ev[m + '_MPSAer']['exact'][p]:.2f}"
                                       for p in ("2", "4", "6", "8", "10")))

# --------------------------------------------------------------------------- Fig. 8: three landscapes
def branch_npy(name):
    raw = subprocess.run(["git", "-C", REPO, "show", f"origin/adding_landscape_results:figures/landscape_lr_mps_pp_qpu/"
                          f"{name}.npy"], capture_output=True, check=True).stdout
    return np.load(io.BytesIO(raw))
L = {"mps": branch_npy("energy_mps_sat_144"), "pp": branch_npy("energy_pp_144"),
     "qpu": branch_npy("energy_qpu_144").reshape(25, 25)}
dbs, dgs = np.linspace(0.1, 2.0, 25), np.linspace(0.1, 2.5, 25)     # rows: delta beta; columns: delta gamma
with open(f"{OUT}/util_land.dat", "w") as fh:
    fh.write("dg db qpu mps pp\n")
    for i in range(25):
        for j in range(25):
            fh.write(f"{dgs[j]:.4f} {dbs[i]:.5f} {L['qpu'][i, j]:.4f} {L['mps'][i, j]:.4f} {L['pp'][i, j]:.4f}\n")
        fh.write("\n")
top = {k: L[k] >= np.percentile(L[k], 90) for k in L}
with open(f"{OUT}/util_land_top.dat", "w") as fh:
    fh.write("x y k\n")
    for kk, k in enumerate(("qpu", "mps", "pp")):
        cs = plt.contour(dgs, dbs, L[k], levels=[np.percentile(L[k], 90)])
        for seg in cs.allsegs[0]:
            for x_, y_ in seg:
                fh.write(f"{x_:.4f} {y_:.4f} {kk}\n")
            fh.write(f"nan nan {kk}\n")
        plt.close("all")
for k in ("mps", "pp", "qpu"):
    i, j = np.unravel_index(np.argmax(L[k]), L[k].shape)
    print(f"Fig. 8, {k}: energies from {L[k].min():.2f} to {L[k].max():.2f}, the maximum at (dg, db) = ({dgs[j]:.3f}, "
          f"{dbs[i]:.3f})")
for k in ("mps", "pp"):
    r = np.corrcoef(L[k].ravel(), L["qpu"].ravel())[0, 1]
    jac = (top[k] & top["qpu"]).sum() / (top[k] | top["qpu"]).sum()
    print(f"  {k} against the device: the correlation {r:.3f}; the top tenths share {(top[k] & top['qpu']).sum()} of "
          f"{top['qpu'].sum()} points, Jaccard {jac:.2f}; the scale {L[k].max() / L['qpu'].max():.2f} of the device's")

# --------------------------------------------------------------------------- Fig. 9: hardware against estimated ratios
doc = fitz.open(f"{PDF}/figure9.pdf"); pg = doc[0]
lab = {str(v): float(v) for v in range(70, 100)}
fy = affine(ticks(pg, lab, "y", (0, 60, 60, 310)))
fxs = [affine(ticks(pg, lab, "x", (60, 315, 440, 340))), affine(ticks(pg, lab, "x", (470, 315, 850, 340)))]
big, small = {}, {}
for d in pg.get_drawings():
    m = TOL.get(rgb(d.get("fill")))
    r = d["rect"]
    if m is None or abs(r.width - r.height) > 0.5 or r.width < 9: continue
    ev_ = "PP" if d["items"][0][0] == "re" else "MPS"
    panel = int((r.x0 + r.x1) / 2 > 450)
    pt = (fxs[panel]((r.x0 + r.x1) / 2), fy((r.y0 + r.y1) / 2))
    if abs(r.width - 15) < 0.5: big.setdefault((m, ev_, panel), []).append(pt)
    elif abs(r.width - 10) < 0.5: small.setdefault((m, ev_, panel), []).append(pt)
for k, v in big.items(): assert len(v) == 1, (k, v)
for k, v in small.items(): assert len(v) == 10, (k, len(v))
series = sorted({(m, e) for m, e, _ in big}, key=lambda me: (me[1], METHODS.index(me[0])))
with open(f"{OUT}/util_hw.dat", "w") as fh:
    fh.write("m ev est5 hw5 est10 hw10 du dv\n")
    for m, e in series:
        (a5, h5), (a10, h10) = big[(m, e, 0)][0], big[(m, e, 1)][0]
        fh.write(f"{METHODS.index(m)} {int(e == 'PP')} {a5:.2f} {h5:.2f} {a10:.2f} {h10:.2f} {a10 - a5:.2f} {h10 - h5:.2f}\n")
print("Fig. 9, the means over the ten graphs (estimated, hardware) at p = 5 and p = 10:")
for m, e in series:
    (a5, h5), (a10, h10) = big[(m, e, 0)][0], big[(m, e, 1)][0]
    print(f"  {m:4s}{e:4s} ({a5:.1f}, {h5:.1f}) -> ({a10:.1f}, {h10:.1f}): the estimate {a10 - a5:+.1f}, the hardware "
          f"{h10 - h5:+.1f}; the gap {a5 - h5:.1f} and {a10 - h10:.1f}")
for e in ("MPS", "PP"):
    for panel, p in ((0, 5), (1, 10)):
        pts = np.array([q for (m, e_, pn), v in small.items() if e_ == e and pn == panel for q in v])
        print(f"  {e} p={p}: the correlation of the {len(pts)} points of the instances {np.corrcoef(pts.T)[0, 1]:.2f}; "
              f"the hardware ratios from {pts[:, 1].min():.1f} to {pts[:, 1].max():.1f}, the estimates from "
              f"{pts[:, 0].min():.1f} to {pts[:, 0].max():.1f}")
for panel, p in ((0, 5), (1, 10)):
    hw = [big[(m, e, panel)][0][1] for m, e in series]; es = [big[(m, e, panel)][0][0] for m, e in series]
    print(f"  p={p}: the means of the {len(series)} series: hardware from {min(hw):.1f} to {max(hw):.1f}, estimated from "
          f"{min(es):.1f} to {max(es):.1f}")

# --------------------------------------------------------------------------- Fig. 10: quality against cost
doc = fitz.open(f"{PDF}/figure10.pdf"); pg = doc[0]
labx = {"10" + s: float(10 ** k) for k, s in ((1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"))}
fy = affine(ticks(pg, lab, "y", (40, 20, 80, 310)))
fx = [affine(ticks(pg, labx, "x", (100, 320, 600, 345)), log=True), affine(ticks(pg, labx, "x", (700, 320, 1200, 345)), log=True)]
cost = {}
for d in pg.get_drawings():
    r = d["rect"]
    if abs(r.width - 18) > 0.5 or abs(r.height - 18) > 0.5: continue
    fill, edge = rgb(d.get("fill")), rgb(d.get("color"))
    if fill == (1.0, 1.0, 1.0): m = "FA0"                       # fixed angles without optimisation (hollow)
    else:
        m = TOL[fill]
        if m in ("LR", "TQA") and edge is None: m += "0"         # linear ramp and TQA with the slopes only
    ev_ = "PP" if d["items"][0][0] == "re" else "MPS"
    panel = int((r.x0 + r.x1) / 2 > 614)
    cost[(m, ev_, panel)] = (fx[panel]((r.x0 + r.x1) / 2), fy((r.y0 + r.y1) / 2))
order = ["FA0", "PT", "TQA0", "LR0", "FA", "LR", "TQA", "I", "F", "RTS"]
with open(f"{OUT}/util_cost.dat", "w") as fh:
    fh.write("m t hw\n")
    for i, m in enumerate(order):
        t, h = cost[(m, "PP", 1)]
        fh.write(f"{i} {t:.1f} {h:.2f}\n")
print("Fig. 10, Pauli propagation, (total duration in s, hardware ratio) at p = 5 and p = 10:")
for m in order:
    print(f"  {m:5s} ({cost[(m, 'PP', 0)][0]:.0f}, {cost[(m, 'PP', 0)][1]:.1f})  ({cost[(m, 'PP', 1)][0]:.0f}, "
          f"{cost[(m, 'PP', 1)][1]:.1f})")
print("  with MPS at p = 10: " + ", ".join(f"{m} ({t:.0f}, {h:.1f})" for (m, e, pn), (t, h) in sorted(cost.items())
                                          if e == "MPS" and pn == 1))

# --------------------------------------------------------------------------- Fig. 12: the Pareto frontiers
doc = fitz.open(f"{PDF}/figure12.pdf"); pg = doc[0]
fy = affine(ticks(pg, {str(v): float(v) for v in range(82, 96, 2)}, "y", (20, 30, 45, 300)))
fx = [affine(ticks(pg, {"10−1": 0.1, "100": 1.0, "101": 10.0, "102": 100.0}, "x", (100, 330, 520, 345)), log=True),
      affine(ticks(pg, {"101": 10.0, "102": 100.0, "103": 1000.0}, "x", (560, 330, 1030, 345)), log=True)]
segs = {0: [], 1: []}
for d in pg.get_drawings():
    col = rgb(d.get("color"))
    if d["type"] != "s" or col not in ((0.267, 0.467, 0.667), (0.781, 0.781, 0.781)) or d.get("dashes") != "[] 0": continue
    if abs((d.get("width") or 0) - 2.6) > 0.01 or d["rect"].y0 > 300: continue
    panel = int(d["rect"].x0 > 540)
    pts = [d["items"][0][1]] + [it[2] for it in d["items"]]
    segs[panel].append((int(col[0] > 0.5), [(fx[panel](q.x), fy(q.y)) for q in pts]))
marks = []
for d in pg.get_drawings():
    r = d["rect"]
    if abs(r.width - 10) > 0.3 or abs(r.height - 10) > 0.3 or d["items"][0][0] != "c" or r.y1 > 300: continue
    fill = rgb(d.get("fill"))
    m = {(0.267, 0.467, 0.667): 0, (1.0, 1.0, 1.0): 1, (0.781, 0.781, 0.781): 2}[fill]
    panel = int((r.x0 + r.x1) / 2 > 540)
    x_, y_ = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    inset = (panel == 0 and 380 < x_ < 520 and 225 < y_ < 300) or (panel == 1 and 680 < x_ < 830 and 225 < y_ < 300)
    if not inset: marks.append((panel, fx[panel](x_), fy(y_), m))
with open(f"{OUT}/util_pareto.dat", "w") as fh:
    fh.write("x y k f\n")
    for k in (0, 1):
        for f_, pts in sorted(segs[k], key=lambda s: s[1][0][0]):
            for x_, y_ in pts:
                fh.write(f"{x_:.5g} {y_:.3f} {k} {f_}\n")
            fh.write(f"nan nan {k} {f_}\n")
with open(f"{OUT}/util_pareto_pts.dat", "w") as fh:
    fh.write("k x y m\n")
    for k, x_, y_, m in sorted(marks):
        fh.write(f"{k} {x_:.5g} {y_:.2f} {m}\n")
for k in (0, 1):
    allp = sorted(q for _, pts in segs[k] for q in pts)
    xs = np.array([q[0] for q in allp]); ys = np.array([q[1] for q in allp])
    print(f"Fig. 12, {'no latency' if k == 0 else '6 s a circuit'}: the frontier from ({xs.min():.3g} s, {ys[0]:.2f}%) to "
          f"({xs.max():.3g} s, {ys.max():.2f}%); the best ratio first at {xs[np.argmax(ys >= ys.max() - 0.02)]:.3g} s; "
          f"90% first at {xs[np.argmax(ys >= 89.98)]:.3g} s")
    runs, cur = [], None
    for f_, pts in sorted(segs[k], key=lambda s: s[1][0][0]):
        if cur and cur[0] == f_: cur[2] = pts[-1][0]
        else:
            cur = [f_, pts[0][0], pts[-1][0]]; runs.append(cur)
    print("  the families along the frontier: " + ", ".join(f"{['FA', 'PT'][f_]} {a:.3g}-{b:.3g} s" for f_, a, b in runs))
    print("  marked: " + ", ".join(f"({x_:.3g} s, {y_:.2f}%, {['FA*', 'FA+', 'PT'][m]})" for kk, x_, y_, m in sorted(marks)
                                   if kk == k))

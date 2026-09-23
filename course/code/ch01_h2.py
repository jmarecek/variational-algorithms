"""H2 in the minimal (STO-3G) basis with PySCF: potential-energy curves and mixing angle, for Lecture 1.
Writes figures/h2/h2_curves.dat (columns R E_HF E0 E1 E2 E3, Angstrom and Hartree)."""
import math, functools, operator
if not hasattr(math, "prod"):                       # Python 3.7 compatibility for pyscf
    math.prod = lambda it, start=1: functools.reduce(operator.mul, it, start)
import numpy as np
from pyscf import gto, scf, fci

rows = []
for R in np.arange(0.4, 3.01, 0.1):
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {R:.2f}", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    cis = fci.FCI(mf); cis.nroots = 4
    e, c = cis.kernel()
    rows.append([R, mf.e_tot] + list(np.atleast_1d(e)))
    if abs(R - 0.75) < 1e-9 or abs(R - 1.5) < 1e-9 or abs(R - 3.0) < 1e-9:
        c0 = np.asarray(c[0]) if isinstance(c, (list, tuple)) else np.asarray(c)
        theta = 2 * math.acos(min(1.0, abs(c0[0, 0])))
        print(f"R={R:.2f} A: E_HF={mf.e_tot:.4f}  E_FCI={e[0]:.4f}  HF weight={c0[0,0]**2:.3f}  theta={theta:.2f} rad")
np.savetxt("figures/h2/h2_curves.dat", np.array(rows), header="R E_HF E0 E1 E2 E3", comments="", fmt="%.6f")

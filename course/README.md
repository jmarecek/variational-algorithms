# Variational Methods in Quantum Optimization — course sources

Lecture notes and slides share one LaTeX source per chapter, in the style of the
quantum-computing course this was derived from (`../template`). Each chapter is a
`subfiles` document; the same file compiles to a handout (article mode via
`beamerarticle`) and to slides (beamer).

## Layout

| File | Purpose |
|---|---|
| `handoutsbeamer.tex` | master for the full handout (`amsbook` + `beamerarticle`) |
| `overheadsbeamer.tex` | master for the full slide deck (`beamer`) |
| `ch01_motivation.tex` … `ch08_warmstart.tex` | the eight lectures (see below) |
| `coursemacros.tex` | macros shared by all chapters: `\figslide`, `\chref`, listings style |
| `variational.bib` | bibliography, **generated** by `makebib.py` from the source papers' `.bib` files |
| `makebib.py` | collects every `\cite` key in `ch*.tex`, looks it up in the source bibliographies, adds hand-written entries for papers that only had `.bbl` files, reports missing keys |
| `build.sh` | `./build.sh` (full handout + deck), `./build.sh ch01` (one chapter, both modes), `./build.sh all` |
| `code/ch01_maxcut_examples.py` | Gurobi / cvxpy+SCS / Qiskit example for Lecture 1; writes `figures/code/*.pdf` |
| `code/ch01_h2.py` | PySCF: H₂ potential-energy curves and mixing angle for Lecture 1; writes `figures/h2/h2_curves.dat` |
| `figures/` | figures copied from the source papers (`numerics`, `warmstart`, `survey`, `annealing`, `itercomp`, `vqa`, `bittel`) plus generated ones (`gw`, `code`, `h2`); many figures in Lecture 1 are TikZ/pgfplots drawn in the source |
| other `.tex`, `.bst`, `.ist` | infrastructure copied unchanged from the template (`declarns.tex`, `macros.tex`, …) |

Build artefacts go to `build/`; the PDFs are copied next to the sources.

## Lectures and their sources (syllabus of 2026-09-23)

1. `ch01_perspectives` — Two perspectives: molecular energy levels (`../chemistry`, Zhang–Wang–Johnson) versus MaxCut; organisation; the Goemans–Williamson relaxation and rounding derived in full; hands-on MaxCut with Gurobi/cvxpy/Qiskit (`code/`); cold-start evidence from `../numerics`, warm-start evidence from `../warmstart`.
2. `ch02_eqa` — The exponential quantum advantage hypothesis (Lee et al. 2023; `../survey` on heuristics; `../chemistry`).
3. `ch03_inapproximability` — Inapproximability and the Unique Games Conjecture (`../survey`, appendices of `../warmstart`, statement of Bittel–Kliesch).
4. `ch04_annealing` — Adiabatic theorem and speedup via annealing (`../annealing`).
5. `ch05_vqe` — Variational quantum eigensolver (template VQA chapter; `../chemistry` for VQE as state preparation).
6. `ch06_qaoa` — QAOA, MaxCut encoding, depth one, and the elementary convergence proof (`../convergence`).
7. `ch07_psr` — Parameter-shift rule and derivative-free estimators (`../iterationcomplexity`, background).
8. `ch08_iterationcomplexity` — Iteration complexity with biased evaluations (`../iterationcomplexity`).
9. `ch09_periteration` — Per-iteration complexity: shots, circuits, simulators, latency; the NP-hardness proof of `../Bittel` (one-page sketch); angle setting at utility scale (`../numerics`).
10. `ch10_ws_rounded` — Warm starts from a GW cut, WS-RQAOA (`../warmstart`).
11. `ch11_ws_continuous` — Warm starts from a continuous relaxation (`../warmstart`).
12. `ch12_vqls` — Variational quantum linear solver, and the early-fault-tolerant alternative (`../chemistry`).
13. `ch13_portfolio` — Markowitz portfolios: levels 1–3 (`../survey`), QUBO encoding, warm-started QAOA results (`../warmstart`).

## Conventions

* `\ona{…}` handout-only text, `\onp{…}` slide-only text, `\ft{…}` frame title (from the template).
* `\figslide[<slide options>]{file}{caption}{label}` shows a figure full-frame on the slide and as a captioned float in the handout.
* `\chref{C.xxx}` refers to a chapter; it falls back to the fixed chapter number when a chapter is compiled on its own. `\Chap`/`\Chaps` print "Chapter(s)" in the handout and "Lecture(s)" in the slides.
* Every chapter ends with a literal `\begin{frame}[allowframebreaks]\ft{References}\biblio\end{frame}`, which prints that chapter's references in a standalone build and a pointer to the deck's bibliography in the full deck (beamer's `ignorenonframetext` needs a literal frame).
* To rebuild the bibliography after adding citations: `python3 makebib.py` (add hand entries to `HAND` in the script if it reports missing keys).

## TODO

* `\courseCode` in `handoutsbeamer.tex` is empty.
* Course organisation (team, assessment) was copied from `../organization.tex` of the quantum optimal control course; times and room updated to Thursdays 9:15–10:45 and 11:00–12:30, KN:A-224. The teaching assistant is Ruben Karapetyan.

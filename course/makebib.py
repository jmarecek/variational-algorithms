#!/usr/bin/env python3
"""Assemble variational.bib from the bibliographies of the source papers.

Collects every \\cite key used in ch*.tex, looks it up (in order) in
../template/jakubs.bib, ../numerics/refs.bib, ../iterationcomplexity/refs.bib,
../survey/references.bib, ../chemistry/ref.bib, ../eqa/ref.bib, adds a few hand-written entries for papers that have no
.bib file (only .bbl), and reports keys that are still missing.
"""
import re, glob, sys

SOURCES = ["../template/jakubs.bib", "../numerics/refs.bib",
           "../iterationcomplexity/refs.bib", "../survey/references.bib", "../chemistry/ref.bib", "../eqa/ref.bib"]

def parse_bib(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", txt):
        if m.group(1).lower() in ("comment", "preamble", "string"):
            continue
        # find matching closing brace
        i = m.end(); depth = 1
        while i < len(txt) and depth > 0:
            if txt[i] == "{": depth += 1
            elif txt[i] == "}": depth -= 1
            i += 1
        entries.setdefault(m.group(2), txt[m.start():i])
    return entries

def numerics_authors():
    txt = open("../numerics/main_v2.tex", encoding="utf-8").read()
    names = re.findall(r"^\\author\{(.*)\}\s*$", txt, re.M)
    return " and ".join(names)

HAND = {
"OAI:The-Unique-Games-Theorem-September-23-2026": '@misc{OAI:The-Unique-Games-Theorem-September-23-2026,\n  author = {{OpenAI}},\n  title = {{The Unique Games Theorem}},\n  howpublished = {OpenAI Math Release preprint\n                  \\href{https://github.com/openai/math/blob/main/preprints/The-Unique-Games-Theorem-September-23-2026/paper.pdf}{OAI:The-Unique-Games-Theorem-September-23-2026}},\n  year = {2026}\n}',
"karp1972reducibility": '@incollection{karp1972reducibility,\n  title={Reducibility among combinatorial problems},\n  author={Karp, Richard M.},\n  booktitle={Complexity of Computer Computations},\n  editor={Miller, Raymond E. and Thatcher, James W.},\n  publisher={Plenum Press},\n  address={New York},\n  pages={85--103},\n  year={1972},\n  doi={10.1007/978-1-4684-2001-2_9}\n}',
"rendl2010solving": '@article{rendl2010solving,\n  title={Solving {Max-Cut} to optimality by intersecting semidefinite and polyhedral relaxations},\n  author={Rendl, Franz and Rinaldi, Giovanni and Wiegele, Angelika},\n  journal={Mathematical Programming},\n  volume={121},\n  number={2},\n  pages={307--335},\n  year={2010},\n  doi={10.1007/s10107-008-0235-8},\n  note={The BiqMac solver, \\url{https://biqmac.aau.at/}}\n}',
"krislock2017biqcrunch": "@article{krislock2017biqcrunch,\n  title={{BiqCrunch}: a semidefinite branch-and-bound method for solving binary quadratic problems},\n  author={Krislock, Nathan and Malick, J{\\'e}r{\\^o}me and Roupin, Fr{\\'e}d{\\'e}ric},\n  journal={ACM Transactions on Mathematical Software},\n  volume={43},\n  number={4},\n  pages={1--23},\n  year={2017},\n  doi={10.1145/3005345},\n  note={\\url{https://biqcrunch.lipn.univ-paris13.fr/}}\n}",
"gusmeroli2022biqbin": '@article{gusmeroli2022biqbin,\n  title={{BiqBin}: a parallel branch-and-bound solver for binary quadratic problems with linear constraints},\n  author={Gusmeroli, Nicol{\\`o} and Hrga, Timotej and Lu{\\v{z}}ar, Borut and Povh, Janez and Siebenhofer, Melanie and Wiegele, Angelika},\n  journal={ACM Transactions on Mathematical Software},\n  volume={48},\n  number={2},\n  pages={1--31},\n  year={2022},\n  doi={10.1145/3514039},\n  note={arXiv:2009.06240}\n}',
"hrga2021madam": '@article{hrga2021madam,\n  title={{MADAM}: a parallel exact solver for max-cut based on semidefinite programming and {ADMM}},\n  author={Hrga, Timotej and Povh, Janez},\n  journal={Computational Optimization and Applications},\n  volume={80},\n  number={2},\n  pages={347--375},\n  year={2021},\n  doi={10.1007/s10589-021-00310-6},\n  note={arXiv:2010.07839}\n}',
"odonnell2014analysis": "@book{odonnell2014analysis,\n  title={Analysis of Boolean Functions},\n  author={O'Donnell, Ryan},\n  publisher={Cambridge University Press},\n  address={Cambridge},\n  year={2014},\n  doi={10.1017/CBO9781139814782},\n  note={arXiv:2105.10386}\n}",
"arora2008unique": '@inproceedings{arora2008unique,\n  title={Unique games on expanding constraint graphs are easy},\n  author={Arora, Sanjeev and Khot, Subhash A. and Kolla, Alexandra and Steurer, David and Tulsiani, Madhur and Vishnoi, Nisheeth K.},\n  booktitle={Proceedings of the 40th ACM Symposium on Theory of Computing},\n  pages={21--28},\n  year={2008},\n  doi={10.1145/1374376.1374380}\n}',
"yannakakis1978node": '@inproceedings{yannakakis1978node,\n  title={Node- and edge-deletion {NP}-complete problems},\n  author={Yannakakis, Mihalis},\n  booktitle={Proceedings of the Tenth Annual ACM Symposium on Theory of Computing},\n  pages={253--264},\n  year={1978},\n  doi={10.1145/800133.804355}\n}',
"berman1999some": '@inproceedings{berman1999some,\n  title={On some tighter inapproximability results},\n  author={Berman, Piotr and Karpinski, Marek},\n  booktitle={Automata, Languages and Programming (ICALP 1999)},\n  series={Lecture Notes in Computer Science},\n  volume={1644},\n  pages={200--209},\n  year={1999},\n  doi={10.1007/3-540-48523-6_17}\n}',
"halperin2004max": '@article{halperin2004max,\n  title={{MAX CUT} in cubic graphs},\n  author={Halperin, Eran and Livnat, Dror and Zwick, Uri},\n  journal={Journal of Algorithms},\n  volume={53},\n  number={2},\n  pages={169--185},\n  year={2004},\n  doi={10.1016/j.jalgor.2004.06.001}\n}',
"brakensiek2024tight": '@inproceedings{brakensiek2024tight,\n  title={Tight approximability of {MAX} 2-{SAT} and relatives, under {UGC}},\n  author={Brakensiek, Joshua and Huang, Neng and Zwick, Uri},\n  booktitle={Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)},\n  pages={1328--1344},\n  year={2024},\n  doi={10.1137/1.9781611977912.53}\n}',
"lewin2002improved": '@inproceedings{lewin2002improved,\n  title={Improved rounding techniques for the {MAX} 2-{SAT} and {MAX} {DI-CUT} problems},\n  author={Lewin, Michael and Livnat, Dror and Zwick, Uri},\n  booktitle={Integer Programming and Combinatorial Optimization (IPCO 2002)},\n  series={Lecture Notes in Computer Science},\n  volume={2337},\n  pages={67--82},\n  year={2002},\n  doi={10.1007/3-540-47867-1_6}\n}',
"austrin2007balanced": '@inproceedings{austrin2007balanced,\n  title={Balanced {Max} 2-{Sat} might not be the hardest},\n  author={Austrin, Per},\n  booktitle={Proceedings of the 39th ACM Symposium on Theory of Computing},\n  pages={189--197},\n  year={2007},\n  doi={10.1145/1250790.1250818}\n}',
"dinur2014analytical": '@inproceedings{dinur2014analytical,\n  title={Analytical approach to parallel repetition},\n  author={Dinur, Irit and Steurer, David},\n  booktitle={Proceedings of the 46th ACM Symposium on Theory of Computing},\n  pages={624--633},\n  year={2014},\n  doi={10.1145/2591796.2591884}\n}',
"zuckerman2007linear": '@article{zuckerman2007linear,\n  title={Linear degree extractors and the inapproximability of max clique and chromatic number},\n  author={Zuckerman, David},\n  journal={Theory of Computing},\n  volume={3},\n  number={1},\n  pages={103--128},\n  year={2007},\n  doi={10.4086/toc.2007.v003a006}\n}',
"christofides1976worst": '@techreport{christofides1976worst,\n  title={Worst-case analysis of a new heuristic for the travelling salesman problem},\n  author={Christofides, Nicos},\n  institution={Graduate School of Industrial Administration, Carnegie Mellon University},\n  number={388},\n  year={1976},\n  note={Reprinted in Operations Research Forum 3:20, 2022, doi:10.1007/s43069-021-00101-z}\n}',
"karlin2021slightly": '@inproceedings{karlin2021slightly,\n  title={A (slightly) improved approximation algorithm for metric {TSP}},\n  author={Karlin, Anna R. and Klein, Nathan and Oveis Gharan, Shayan},\n  booktitle={Proceedings of the 53rd ACM Symposium on Theory of Computing},\n  pages={32--45},\n  year={2021},\n  doi={10.1145/3406325.3451009}\n}',
"karpinski2015new": '@article{karpinski2015new,\n  title={New inapproximability bounds for {TSP}},\n  author={Karpinski, Marek and Lampis, Michael and Schmied, Richard},\n  journal={Journal of Computer and System Sciences},\n  volume={81},\n  number={8},\n  pages={1665--1677},\n  year={2015},\n  doi={10.1016/j.jcss.2015.06.003}\n}',
"arora2008euclidean": '@article{arora2008euclidean,\n  title={Euclidean distortion and the sparsest cut},\n  author={Arora, Sanjeev and Lee, James R. and Naor, Assaf},\n  journal={Journal of the American Mathematical Society},\n  volume={21},\n  number={1},\n  pages={1--21},\n  year={2008},\n  doi={10.1090/S0894-0347-07-00573-5}\n}',
"mossel2010noise": "@article{mossel2010noise,\n  title={Noise stability of functions with low influences: invariance and optimality},\n  author={Mossel, Elchanan and O'Donnell, Ryan and Oleszkiewicz, Krzysztof},\n  journal={Annals of Mathematics},\n  volume={171},\n  number={1},\n  pages={295--341},\n  year={2010},\n  doi={10.4007/annals.2010.171.295}\n}",
"khot2008vertex": '@article{khot2008vertex,\n  title={Vertex cover might be hard to approximate to within $2-\\varepsilon$},\n  author={Khot, Subhash and Regev, Oded},\n  journal={Journal of Computer and System Sciences},\n  volume={74},\n  number={3},\n  pages={335--349},\n  year={2008},\n  doi={10.1016/j.jcss.2007.06.019}\n}',
"arora2015subexponential": '@article{arora2015subexponential,\n  title={Subexponential algorithms for unique games and related problems},\n  author={Arora, Sanjeev and Barak, Boaz and Steurer, David},\n  journal={Journal of the ACM},\n  volume={62},\n  number={5},\n  pages={1--25},\n  year={2015},\n  doi={10.1145/2775105}\n}',
"raghavendra2010graph": '@inproceedings{raghavendra2010graph,\n  title={Graph expansion and the unique games conjecture},\n  author={Raghavendra, Prasad and Steurer, David},\n  booktitle={Proceedings of the 42nd ACM Symposium on Theory of Computing},\n  pages={755--764},\n  year={2010},\n  doi={10.1145/1806689.1806792}\n}',
"raghavendra2012reductions": '@inproceedings{raghavendra2012reductions,\n  title={Reductions between expansion problems},\n  author={Raghavendra, Prasad and Steurer, David and Tulsiani, Madhur},\n  booktitle={2012 IEEE 27th Conference on Computational Complexity},\n  pages={64--73},\n  year={2012},\n  doi={10.1109/CCC.2012.43}\n}',
"singer2011angular": '@article{singer2011angular,\n  title={Angular synchronization by eigenvectors and semidefinite programming},\n  author={Singer, Amit},\n  journal={Applied and Computational Harmonic Analysis},\n  volume={30},\n  number={1},\n  pages={20--36},\n  year={2011},\n  doi={10.1016/j.acha.2010.02.001}\n}',
"murty1987some": '@article{murty1987some,\n  title={Some {NP}-complete problems in quadratic and nonlinear programming},\n  author={Murty, Katta G. and Kabadi, Santosh N.},\n  journal={Mathematical Programming},\n  volume={39},\n  number={2},\n  pages={117--129},\n  year={1987},\n  doi={10.1007/BF02592948}\n}',
"eriksen2020ground": '@article{eriksen2020ground,\n  title={The ground state electronic energy of benzene},\n  author={Eriksen, Janus J. and Anderson, Tyler A. and Deustua, J. Emiliano and Ghanem, Khaldoon and Hait, Diptarka and Hoffmann, Mark R. and Lee, Seunghoon and Levine, Daniel S. and Magoulas, Ilias and Shen, Jun and Tubman, Norman M. and Whaley, K. Birgitta and Xu, Enhua and Yao, Yuan and Zhang, Ning and Alavi, Ali and Chan, Garnet Kin-Lic and Head-Gordon, Martin and Liu, Wenjian and Piecuch, Piotr and Sharma, Sandeep and Ten-no, Seiichiro L. and Umrigar, C. J. and Gauss, J{\\"u}rgen},\n  journal={The Journal of Physical Chemistry Letters},\n  volume={11},\n  number={20},\n  pages={8922--8929},\n  year={2020},\n  doi={10.1021/acs.jpclett.0c02621}\n}',
"feynman1982simulating": '@article{feynman1982simulating,\n  title={Simulating physics with computers},\n  author={Feynman, Richard P.},\n  journal={International Journal of Theoretical Physics},\n  volume={21},\n  number={6--7},\n  pages={467--488},\n  year={1982},\n  doi={10.1007/BF02650179}\n}',
"lloyd1996universal": '@article{lloyd1996universal,\n  title={Universal quantum simulators},\n  author={Lloyd, Seth},\n  journal={Science},\n  volume={273},\n  number={5278},\n  pages={1073--1078},\n  year={1996},\n  doi={10.1126/science.273.5278.1073}\n}',
"ogorman2022intractability": "@article{ogorman2022intractability,\n  title={Intractability of electronic structure in a fixed basis},\n  author={O'Gorman, Bryan and Irani, Sandy and Whitfield, James and Fefferman, Bill},\n  journal={PRX Quantum},\n  volume={3},\n  number={2},\n  pages={020322},\n  year={2022},\n  doi={10.1103/PRXQuantum.3.020322}\n}",
"omalley2016scalable": "@article{omalley2016scalable,\n  title={Scalable quantum simulation of molecular energies},\n  author={O'Malley, Peter J. J. and Babbush, Ryan and Kivlichan, Ian D. and Romero, Jonathan and McClean, Jarrod R. and Barends, Rami and Kelly, Julian and Roushan, Pedram and Tranter, Andrew and Ding, Nan and others},\n  journal={Physical Review X},\n  volume={6},\n  number={3},\n  pages={031007},\n  year={2016}\n}",
"zhang2022computing": "@article{zhang2022computing,\n  title={Computing ground state properties with early fault-tolerant quantum computers},\n  author={Zhang, Ruizhe and Wang, Guoming and Johnson, Peter},\n  journal={Quantum},\n  volume={6},\n  pages={761},\n  year={2022},\n  doi={10.22331/q-2022-07-11-761}\n}",
"lee2023evaluating": "@article{lee2023evaluating,\n  title={Evaluating the evidence for exponential quantum advantage in ground-state quantum chemistry},\n  author={Lee, Seunghoon and Lee, Joonho and Zhai, Huanchen and Tong, Yu and Dalzell, Alexander M. and Kumar, Ashutosh and Helms, Phillip and Gray, Johnnie and Cui, Zhi-Hao and Liu, Wenyuan and Kastoryano, Michael and Babbush, Ryan and Preskill, John and Reichman, David R. and Campbell, Earl T. and Valeev, Edward F. and Lin, Lin and Chan, Garnet Kin-Lic},\n  journal={Nature Communications},\n  volume={14},\n  pages={1952},\n  year={2023}\n}",
"bravoprieto2023vqls": "@article{bravoprieto2023vqls,\n  title={Variational quantum linear solver},\n  author={Bravo-Prieto, Carlos and LaRose, Ryan and Cerezo, M. and Subasi, Yigit and Cincio, Lukasz and Coles, Patrick J.},\n  journal={Quantum},\n  volume={7},\n  pages={1188},\n  year={2023}\n}",
"kandala2017hardware": "@article{kandala2017hardware,\n  title={Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets},\n  author={Kandala, Abhinav and Mezzacapo, Antonio and Temme, Kristan and Takita, Maika and Brink, Markus and Chow, Jerry M. and Gambetta, Jay M.},\n  journal={Nature},\n  volume={549},\n  number={7671},\n  pages={242--246},\n  year={2017}\n}",
"kristaly2010variational": "@book{kristaly2010variational,\n  title={Variational Principles in Mathematical Physics, Geometry, and Economics: Qualitative Analysis of Nonlinear Equations and Unilateral Problems},\n  author={Krist{\\'a}ly, Alexandru and R{\\u{a}}dulescu, Vicen{\\c{t}}iu D. and Varga, Csaba},\n  series={Encyclopedia of Mathematics and its Applications},\n  volume={136},\n  publisher={Cambridge University Press},\n  address={Cambridge, UK},\n  year={2010},\n  note={ISBN 978-0-521-11782-1}\n}",
"Hodson2019": "@misc{Hodson2019,\n  title={Portfolio rebalancing experiments using the quantum alternating operator ansatz},\n  author={Hodson, Mark and Ruck, Brendan and Ong, Hugh and Garvin, David and Dulman, Stefan},\n  howpublished={arXiv:1911.05296},\n  year={2019}\n}",
"marecek2026angles": "@article{marecek2026angles,\n  title={Setting angles in quantum approximate optimization at utility-scale},\n  author={%s},\n  journal={Preprint},\n  year={2026}\n}" % numerics_authors(),
"nagaj2012quantum": "@article{nagaj2012quantum,\n  title={Quantum speedup by quantum annealing},\n  author={Somma, Rolando D. and Nagaj, Daniel and Kieferov{\\'a}, M{\\'a}ria},\n  journal={Physical Review Letters},\n  volume={109},\n  number={5},\n  pages={050501},\n  year={2012},\n  publisher={APS}\n}",
"aspman2024quantum": "@misc{aspman2024quantum,\n  title={Quantum Computing via Randomized Algorithms},\n  author={Aspman, Johannes and Korpas, Georgios and Mare{\\v c}ek, Jakub},\n  howpublished={Lecture notes, Czech Technical University in Prague},\n  year={2026}\n}",
"Teufel2001": "@article{Teufel2001,\n  title={A note on the adiabatic theorem without gap condition},\n  author={Teufel, Stefan},\n  journal={Letters in Mathematical Physics},\n  volume={58},\n  number={3},\n  pages={261--266},\n  year={2001},\n  publisher={Springer}\n}",
"Kato1995": "@book{Kato1995,\n  title={Perturbation Theory for Linear Operators},\n  author={Kato, Tosio},\n  year={1995},\n  edition={2},\n  publisher={Springer},\n  address={Berlin}\n}",
"charikar2004maximizing": "@inproceedings{charikar2004maximizing,\n  title={Maximizing quadratic programs: extending {G}rothendieck's inequality},\n  author={Charikar, Moses and Wirth, Anthony},\n  booktitle={45th Annual IEEE Symposium on Foundations of Computer Science},\n  pages={54--60},\n  year={2004},\n  organization={IEEE}\n}",
"Markowitz1952": "@article{Markowitz1952,\n  title={Portfolio selection},\n  author={Markowitz, Harry},\n  journal={The Journal of Finance},\n  volume={7},\n  number={1},\n  pages={77--91},\n  year={1952}\n}",
"poljak1995recipe": "@article{poljak1995recipe,\n  title={A recipe for semidefinite relaxation for (0,1)-quadratic programming},\n  author={Poljak, Svatopluk and Rendl, Franz and Wolkowicz, Henry},\n  journal={Journal of Global Optimization},\n  volume={7},\n  number={1},\n  pages={51--73},\n  year={1995},\n  publisher={Springer}\n}",
"khot2002power": "@inproceedings{khot2002power,\n  title={On the power of unique 2-prover 1-round games},\n  author={Khot, Subhash},\n  booktitle={Proceedings of the 34th Annual ACM Symposium on Theory of Computing},\n  pages={767--775},\n  year={2002}\n}",
"Bravyi2019": "@article{Bravyi2019,\n  title={Obstacles to variational quantum optimization from symmetry protection},\n  author={Bravyi, Sergey and Kliesch, Alexander and Koenig, Robert and Tang, Eugene},\n  journal={Physical Review Letters},\n  volume={125},\n  number={26},\n  pages={260505},\n  year={2020},\n  publisher={APS}\n}",
"Wang2018": "@article{Wang2018,\n  title={Quantum approximate optimization algorithm for {MaxCut}: A fermionic view},\n  author={Wang, Zhihui and Hadfield, Stuart and Jiang, Zhang and Rieffel, Eleanor G.},\n  journal={Physical Review A},\n  volume={97},\n  number={2},\n  pages={022304},\n  year={2018},\n  publisher={APS}\n}",
}

def main():
    keys = set()
    for f in sorted(glob.glob("ch*.tex")):
        txt = open(f, encoding="utf-8").read()
        for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", txt):
            for k in m.group(1).split(","):
                k = k.strip()
                if k: keys.add(k)
    bibs = [parse_bib(p) for p in SOURCES]
    out, missing = [], []
    for k in sorted(keys):
        for b in bibs:
            if k in b:
                out.append(b[k]); break
        else:
            if k in HAND: out.append(HAND[k])
            else: missing.append(k)
    with open("variational.bib", "w", encoding="utf-8") as fh:
        fh.write("% Generated by makebib.py from the bibliographies of the source papers. Do not edit; edit makebib.py.\n\n")
        fh.write("\n\n".join(out) + "\n")
    print(f"{len(out)} entries written, {len(missing)} missing:", *missing)

if __name__ == "__main__":
    main()

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
"eriksen2020ground": "@article{eriksen2020ground,\n  title={The ground state electronic energy of benzene},\n  author={Eriksen, Janus J. and Anderson, Tyler A. and Deustua, J. Emiliano and Ghanem, Khaldoon and Hait, Diptarka and Hoffmann, Mark R. and Lee, Seunghoon and Levine, Daniel S. and Magoulas, Ilias and Shen, Jun and Tubman, Norman M. and Whaley, K. Birgitta and Xu, Enhua and Yao, Yuan and Zhang, Ning and Alavi, Ali and Chan, Garnet Kin-Lic and Head-Gordon, Martin and Liu, Wenjian and Piecuch, Piotr and Sharma, Sandeep and Ten-no, Seiichiro L. and Umrigar, C. J. and Gauss, J{\\\"u}rgen},\n  journal={The Journal of Physical Chemistry Letters},\n  volume={11},\n  pages={8922},\n  year={2020},\n  note={arXiv:2008.02678}\n}",
"feynman1982simulating": "@article{feynman1982simulating,\n  title={Simulating physics with computers},\n  author={Feynman, Richard P.},\n  journal={International Journal of Theoretical Physics},\n  volume={21},\n  number={6--7},\n  pages={467--488},\n  year={1982}\n}",
"lloyd1996universal": "@article{lloyd1996universal,\n  title={Universal quantum simulators},\n  author={Lloyd, Seth},\n  journal={Science},\n  volume={273},\n  number={5278},\n  pages={1073--1078},\n  year={1996}\n}",
"ogorman2022intractability": "@article{ogorman2022intractability,\n  title={Intractability of electronic structure in a fixed basis},\n  author={O'Gorman, Bryan and Irani, Sandy and Whitfield, James and Fefferman, Bill},\n  journal={PRX Quantum},\n  volume={3},\n  number={2},\n  pages={020322},\n  year={2022}\n}",
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

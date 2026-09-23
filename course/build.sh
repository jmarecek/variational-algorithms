#!/bin/bash
# Build the course.
#   ./build.sh            full handout (handoutsbeamer.pdf) and full slide deck (overheadsbeamer.pdf)
#   ./build.sh chNN       one chapter standalone, as handout (chNN_*.pdf) and as slides (slides_chNN_*.pdf)
#   ./build.sh all        everything
# Auxiliary files go to build/.
set -u
mkdir -p build
run() { # run <jobname> <texfile>
  local job=$1 tex=$2
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build -jobname="$job" "$tex" > "build/$job.log1" 2>&1 || { echo "FAILED: $job (see build/$job.log)"; grep -n -A5 '^! ' "build/$job.log" | head -20; return 1; }
  ( cd build && BIBINPUTS=..: BSTINPUTS=..: bibtex "$job" > "$job.blg1" 2>&1 )
  pdflatex -interaction=nonstopmode -output-directory=build -jobname="$job" "$tex" > /dev/null 2>&1
  pdflatex -interaction=nonstopmode -output-directory=build -jobname="$job" "$tex" > "build/$job.log1" 2>&1
  cp "build/$job.pdf" "$job.pdf"
  printf "%-28s pages %3s  undefined refs %2s  undefined cites %2s  overfull %2s\n" "$job" "$(pdfinfo "$job.pdf" | awk '/Pages/{print $2}')" \
    "$(grep -c 'Reference.*undefined' "build/$job.log")" "$(grep -c 'Citation.*undefined' "build/$job.log")" "$(grep -c 'Overfull' "build/$job.log")"
}
chapter() { # chapter <basename without .tex>
  local ch=$1
  run "$ch" "$ch.tex"
  sed 's/^\\documentclass\[handoutsbeamer.tex\]{subfiles}/\\documentclass[overheadsbeamer.tex]{subfiles}/' "$ch.tex" > "build/slides_$ch.tex"
  cp "build/slides_$ch.tex" "slides_$ch.tex"   # must sit next to the master files for \documentclass[..]{subfiles}
  run "slides_$ch" "slides_$ch.tex"
  rm -f "slides_$ch.tex"
}
case "${1:-full}" in
  full) run handoutsbeamer handoutsbeamer.tex; run overheadsbeamer overheadsbeamer.tex ;;
  all)  run handoutsbeamer handoutsbeamer.tex; run overheadsbeamer overheadsbeamer.tex
        for f in ch[0-9][0-9]_*.tex; do chapter "${f%.tex}"; done ;;
  ch*)  for f in $1*.tex; do chapter "${f%.tex}"; done ;;
esac

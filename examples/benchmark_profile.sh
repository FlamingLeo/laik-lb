#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

PROGNAME="${1:-md-lb}"
NTASKS_LIST=(2 4 8 16 32 64)
ALGOS=(rcb rcbincr hilbert gilbert)
FREQS=(150)

# make a unique destination name if it already exists
unique_name() {
  local base="$1"
  if [[ ! -e "$base" ]]; then
    printf "%s" "$base"
    return
  fi
  local i=1
  while [[ -e "${base}-$i" ]]; do
    ((i++))
  done
  printf "%s-%d" "$base" "$i"
}

echo "Starting runs with program: ${PROGNAME}"

# iterate combinations
for nt in "${NTASKS_LIST[@]}"; do
  for algo in "${ALGOS[@]}"; do
    for freq in "${FREQS[@]}"; do
      mkdir -p lbviz/json
      sleep 1

      combo_tag="${PROGNAME}_nt${nt}_${algo}_f${freq}"
      echo "============================================================"
      echo "Running: NTASKS=${nt}, ALGO=${algo}, FREQ=${freq}"
      echo "Command: mpirun -n ${nt} ./${PROGNAME} -o -p -a ${algo} -n ${freq}"
      
      # run mpirun
      if mpirun -n "${nt}" ./"${PROGNAME}" -o -p -a "${algo}"; then
        echo "Run finished (exit code 0)."
      else
        echo "Run FAILED (non-zero exit). Continuing to next combo." >&2
      fi

      # sleep 1 second after run
      sleep 1

      # rename produced folder
      produced="lbviz/json"
      if [[ -d "${produced}" ]]; then
        dest_base="lbviz/json-${combo_tag}"
        dest=$(unique_name "${dest_base}")
        echo "Renaming ${produced} -> ${dest}"
        mv "${produced}" "${dest}"
        # sleep 1 second after rename
        sleep 1
      else
        echo "Warning: expected folder '${produced}' not found after run. Skipping rename." >&2
      fi

      echo "Completed combo: ${combo_tag}"
      echo
    done
  done
done

mkdir -p lbviz/json

echo "Done."

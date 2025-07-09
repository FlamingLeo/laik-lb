#!/bin/bash
set -euo pipefail

# read command line arguments to replace them in actual program call
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <ntasks> <progname> [opts...]"
  exit 1
fi

n="$1"
shift
cmd="$1"
shift
args=( "$@" )

# remove old perf files
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}"
./clean_perf.sh

# generate new perf data and convert it to firefox profiler format
cd "${SCRIPT_DIR}/../examples"
mkdir -p perf
sudo perf record -g -F 999 mpirun -n "$n" --allow-run-as-root "./$cmd" "${args[@]}"

sudo chown $USER perf.data
perf script -F +pid > firefox.perf

# housekeeping
mv perf.* perf
mv *.perf perf
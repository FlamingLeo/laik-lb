#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}/.."

# delete everything perf-related (redundancy for .data or .old files, incl. perf directory)
find . -name 'perf.*' -type f -delete
find . -name '*.perf' -type f -delete
find . -name '*.data' -type f -delete
find . -name '*.old' -type f -delete
find . -name '*perf*' -type d -delete
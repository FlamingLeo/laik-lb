#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}/../examples/lbviz"
rm -f *.png
rm -f *.txt
rm -f *.json
rm -f *.svg